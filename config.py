"""
FlyBrain-HalfLife: Configuration Module
MaleCNS v1.0 & FlyWire Connectome Simulation Hyperparameters
Includes Low-Latency I/O, AGC Current Normalization, and Deadlock Prevention
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import os

@dataclass
class SimulationConfig:
    dt: float = 1.0                    # Simulation time step (ms)
    v_rest: float = -70.0              # Resting membrane potential (mV)
    v_reset: float = -75.0             # Post-spike reset potential (mV)
    v_thresh: float = -50.0            # Spiking threshold potential (mV)
    tau_m: float = 10.0                # Membrane time constant (ms)
    refractory_period: int = 2         # Refractory period (step count)
    synaptic_decay: float = 0.85       # Synaptic current decay factor
    use_pytorch: bool = True           # Use GPU/CPU PyTorch sparse tensors if available
    device: str = "cpu"                # 'cuda' or 'cpu'
    
    # Deadlock Prevention - Spontaneous Biological Background Noise
    spontaneous_noise_std: float = 2.5 # Synthetic Poisson / Gaussian background current noise (nA)
    spontaneous_bias: float = 1.2      # Basal tonic current keeping neurons active (nA)

@dataclass
class VisionConfig:
    grid_width: int = 60               # Hexagonal / Cartesian ommatidia grid width
    grid_height: int = 60              # Hexagonal / Cartesian ommatidia grid height
    total_ommatidia: int = 3600        # 60x60 photoreceptor array
    r1_r6_weight: float = 0.8          # Luminance channel weight (R1-R6 broad spectrum)
    r8_weight: float = 0.2             # Spectral / color contrast weight (R8 threat/pickup)
    motion_sensitivity: float = 1.5    # T4/T5 directional optic flow gain
    fps_target: int = 30               # Target screen capture and processing FPS
    capture_window_title: str = "Half-Life" # Window title to capture
    
    # Adaptive Gain Control (AGC) - Prevents silence and saturation
    use_agc: bool = True               # Enable AGC
    target_current_mean: float = 12.0  # Target average photoreceptor current (nA)
    target_current_max: float = 28.0   # Saturation current ceiling (nA)
    agc_adaptation_rate: float = 0.05  # AGC adaptation rate

@dataclass
class ConnectomeConfig:
    # Biological Scale Neuron Population Counts
    num_photoreceptors: int = 3600     # R1-R8 retinal input units
    num_optic_lobe: int = 4800         # Lamina, Medulla, Lobula layers
    num_central_complex: int = 1200    # Ellipsoid Body (EB) heading ring & Protocerebral Bridge (PB)
    num_mushroom_body_kc: int = 2400   # Kenyon Cells (Associative memory)
    num_mbon: int = 120                # Mushroom Body Output Neurons
    num_dopaminergic: int = 60         # PPL1 dopaminergic cluster (reward/penalty)
    num_descending: int = 80           # Descending neurons (DN premotor pathways)
    
    # Dedicated Descending Motor Neuron IDs
    dnp20_left_id: int = 0             # DNp20 Left (Steer Left)
    dnp20_right_id: int = 1            # DNp20 Right (Steer Right)
    dnpe017_forward_id: int = 2        # DNpe017 Forward Walking
    dnpe017_attack_id: int = 3         # DNpe017 Attack / Primary Fire
    
    # Graph Sparsity and Random Seed
    sparsity: float = 0.015            # Synaptic connection density
    seed: int = 42

@dataclass
class ReinforcementConfig:
    base_dopamine: float = 0.0         # Baseline dopamine concentration
    learning_rate: float = 0.005       # KC -> MBON STDP synaptic plasticity rate
    damage_penalty: float = -1.5       # Negative dopamine impulse upon taking damage
    kill_reward: float = 2.0           # Positive dopamine impulse upon target kill
    forward_reward: float = 0.1        # Small positive reward for forward exploration
    
    # Obstacle Anti-Stuck & Panic Mode (45 steps = ~1.5s threshold)
    stuck_threshold_steps: int = 45    # Obstacle deadlock detection threshold (~1.5s)
    panic_noise_strength: float = 24.0 # Chaotic burst current injected during panic
    panic_duration_steps: int = 15     # Duration of panic mode escape state (steps)

@dataclass
class MotorConfig:
    turn_threshold: float = 0.03       # DNp20 differential steering threshold
    walk_threshold: float = 0.05       # DNpe017 forward walking threshold
    attack_threshold: float = 0.35     # DNpe017 attack / fire threshold (prevents walking noise fire)
    press_duration: float = 0.06       # Minimum hardware key hold duration (s)
    cooldown_duration: float = 0.08    # Hardware debounce / refractory cooldown (s)
    dry_run: bool = False              # If True, simulate actions without sending OS keystrokes

@dataclass
class TelemetryConfig:
    window_width: int = 1280
    window_height: int = 720
    brain_pane_width: int = 640
    game_pane_width: int = 640
    point_size: int = 3
    glow_decay: float = 0.88
    font_name: str = "Arial"

@dataclass
class Config:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    connectome: ConnectomeConfig = field(default_factory=ConnectomeConfig)
    rl: ReinforcementConfig = field(default_factory=ReinforcementConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    cache_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

DEFAULT_CONFIG = Config()
