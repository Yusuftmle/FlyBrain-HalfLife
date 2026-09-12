"""
Dopamine Neuromodulation & Synaptic Plasticity (PPL1 & KC-MBON STDP)
Simulates Dopaminergic Reinforcement Loop and 2-Second Anti-Stuck Panic Mode
"""
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Tuple
from config import ReinforcementConfig, DEFAULT_CONFIG
from core.connectome import FlyConnectome
from core.lif_engine import LIFEngine

class DopamineController:
    """
    PPL101 Dopaminergic Circuit & Mushroom Body (KC -> MBON) Plasticity.
    - Releases negative dopamine on damage (punishment).
    - Releases positive dopamine on kills/forward progress (reward).
    - Triggers synthetic PANIC / DAMAGE mode and chaotic 180-degree escape spin
      if stationary for 2 seconds (e.g. stuck against a wall).
    """
    def __init__(self, connectome: FlyConnectome, lif_engine: LIFEngine, cfg: ReinforcementConfig = DEFAULT_CONFIG.rl):
        self.cfg = cfg
        self.connectome = connectome
        self.lif_engine = lif_engine
        
        # Neuron population indices
        self.dopamine_indices = connectome.get_dopamine_indices()
        self.kc_indices, self.mbon_indices = connectome.get_kc_mbon_indices()
        
        # Dopamine level dynamics
        self.dopamine_level: float = cfg.base_dopamine
        self.da_decay: float = 0.92
        
        # Fast Obstacle Anti-Stuck & Panic state tracking
        self.stationary_steps: int = 0
        self.panic_counter: int = 0
        self.is_in_panic: bool = False
        self.escape_direction: int = 1
        
        # Statistics
        self.total_rewards: float = 0.0
        self.total_penalties: float = 0.0
        self.plasticity_events: int = 0

    def register_event(self, event_type: str, magnitude: float = 1.0):
        """
        Applies instantaneous dopaminergic perturbation based on game events.
        """
        if event_type == "damage":
            delta = self.cfg.damage_penalty * magnitude
            self.dopamine_level = max(-3.0, self.dopamine_level + delta)
            self.total_penalties += abs(delta)
        elif event_type == "kill":
            delta = self.cfg.kill_reward * magnitude
            self.dopamine_level = min(4.0, self.dopamine_level + delta)
            self.total_rewards += delta
        elif event_type == "forward":
            delta = self.cfg.forward_reward * magnitude
            self.dopamine_level = min(2.0, self.dopamine_level + delta)
            self.total_rewards += delta

    def update(self, motion_flow: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Updates dopamine decay, monitors stationary obstacle state,
        computes PPL1 injection current, and applies 3-factor STDP.
        """
        # 1. Dopamine decay back to baseline
        self.dopamine_level = (self.dopamine_level - self.cfg.base_dopamine) * self.da_decay + self.cfg.base_dopamine

        # 2. Obstacle Deadlock Detection (Stationary against wall)
        if motion_flow < 0.025:
            self.stationary_steps += 1
        else:
            self.stationary_steps = max(0, self.stationary_steps - 2)

        ext_current = np.zeros(self.connectome.total_neurons, dtype=np.float32)

        # 3. Trigger Unstuck Maneuver if Facing Obstacle
        if self.stationary_steps >= self.cfg.stuck_threshold_steps:
            self.is_in_panic = True
            self.panic_counter = self.cfg.panic_duration_steps
            self.dopamine_level = -2.5
            self.stationary_steps = 0
            # Consistently choose an escape direction (1 = Turn Right, -1 = Turn Left)
            self.escape_direction = 1 if np.random.rand() > 0.5 else -1

        if self.panic_counter > 0:
            self.panic_counter -= 1
            self.is_in_panic = True
            
            # Asymmetric burst into Central Complex (EB) steering ring
            cx_offset = self.connectome.region_offsets["central_complex"]
            cx_count = self.connectome.region_counts["central_complex"]
            ext_current[cx_offset : cx_offset + cx_count] += np.random.uniform(12.0, self.cfg.panic_noise_strength, cx_count)

            # Consistently drive ONE turning side away from the wall (NOT alternating each frame)
            motor_indices = self.connectome.get_motor_neuron_indices()
            escape_side = motor_indices["dnp20_right"] if self.escape_direction > 0 else motor_indices["dnp20_left"]
            ext_current[escape_side] += 40.0
            # NOTE: Deliberately do NOT add forward drive here so the fly does not push into the obstacle!
        else:
            self.is_in_panic = False

        # 4. Excite PPL1 Dopaminergic Cluster
        da_current = self.dopamine_level * 12.0
        ext_current[self.dopamine_indices] += da_current

        # 5. Three-Factor STDP Synaptic Plasticity
        if abs(self.dopamine_level) > 0.1:
            self._apply_stdp_update()

        telemetry = {
            "dopamine_level": float(self.dopamine_level),
            "stationary_steps": self.stationary_steps,
            "is_in_panic": self.is_in_panic,
            "escape_direction": self.escape_direction,
            "total_rewards": float(self.total_rewards),
            "total_penalties": float(self.total_penalties),
            "plasticity_events": self.plasticity_events
        }

        return ext_current, telemetry

    def _apply_stdp_update(self):
        """Updates synaptic weights between KC and MBON based on dopamine level."""
        spikes = self.lif_engine.spikes
        kc_fired = spikes[self.kc_indices]
        mbon_fired = spikes[self.mbon_indices]

        active_kc = np.where(kc_fired)[0]
        active_mbon = np.where(mbon_fired)[0]

        if len(active_kc) == 0 or len(active_mbon) == 0:
            return

        delta = self.cfg.learning_rate * (self.dopamine_level - self.cfg.base_dopamine)
        weights = self.connectome.weights
        abs_kc = self.kc_indices[active_kc]
        abs_mbon = self.mbon_indices[active_mbon]

        for src in abs_kc:
            row_start = weights.indptr[src]
            row_end = weights.indptr[src + 1]
            cols = weights.indices[row_start:row_end]
            
            mask = np.isin(cols, abs_mbon)
            if np.any(mask):
                weights.data[row_start:row_end][mask] += delta
                weights.data[row_start:row_end][mask] = np.clip(weights.data[row_start:row_end][mask], 0.01, 4.0)
                self.plasticity_events += 1
