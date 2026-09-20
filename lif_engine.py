"""
lif_engine.py - PyTorch GPU Vectorized DOOMFLY LIF Simulation Engine
Faithfully adapts Alex Wormuth's DOOMFLY / MaleCNS v1.0 biophysical differential equations:
  - Analytic subthreshold integration: tau_m = 20ms, tau_s = 5ms, coupling = (av - ag)/3
  - Resting potential V_rest = -52 mV, Spiking threshold V_th = -45 mV, Reset = -52 mV
  - Synaptic conductance dynamics: g *= ag, delivered via sparse COO matrix multiplication
  - Transmission delay ring buffer (1.8 ms delay queue)
  - 100% Vectorized PyTorch CUDA sparse execution (Zero per-neuron for-loops)
"""
import math
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger("FlyBrain.LIFEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class PyTorchLIFEngine:
    """
    Exact Biophysical DOOMFLY Leaky Integrate-and-Fire Simulation Engine.
    Governed by analytic two-variable differential system:
      dV/dt = -(V - V_rest)/tau_m + I_drive + G * coupling
      dG/dt = -G / tau_s
    All state transitions are vectorized across 100K+ neurons on CUDA GPU.
    """
    def __init__(
        self,
        sparse_weights: Any,
        num_neurons: int,
        dt: float = 0.5,                # Integration step in ms (0.1ms to 1.0ms)
        v_rest: float = -52.0,           # DOOMFLY calibrated resting potential (mV)
        v_reset: float = -52.0,          # DOOMFLY reset potential (mV)
        v_thresh: float = -45.0,         # DOOMFLY firing threshold (mV)
        tau_m: float = 20.0,             # Membrane time constant (ms)
        tau_s: float = 5.0,              # Synaptic conductance time constant (ms)
        transmission_delay_ms: float = 1.8, # Biological conduction delay (ms)
        refractory_ms: float = 2.2,      # Absolute refractory duration (ms)
        spontaneous_noise_std: float = 1.5, # Deadlock prevention baseline noise (nA)
        device: Optional[str] = None
    ):
        self.num_neurons: int = num_neurons
        self.dt: float = dt
        self.v_rest: float = v_rest
        self.v_reset: float = v_reset
        self.v_thresh: float = v_thresh
        self.tau_m: float = tau_m
        self.tau_s: float = tau_s
        self.spontaneous_noise_std: float = spontaneous_noise_std
        
        # DOOMFLY Analytic Exponential Factors
        self.av: float = float(math.exp(-self.dt / self.tau_m))
        self.ag: float = float(math.exp(-self.dt / self.tau_s))
        self.coupling: float = float((self.av - self.ag) / 3.0)
        self.refractory_steps: int = max(1, int(round(refractory_ms / self.dt)))
        
        # Transmission Delay Queue Slots (1.8 ms delay)
        self.delay_slots: int = max(2, int(round(transmission_delay_ms / self.dt)) + 1)
        self.cursor: int = 0
        
        # Hardware Device Selection (CUDA Priority)
        if device is not None:
            self.device_str = device
        elif torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
            
        self.device = torch.device(self.device_str)
        logger.info(f"PyTorchLIFEngine initialized: {self.num_neurons:,} neurons on '{self.device_str.upper()}'.")

        # Compile GPU Sparse Synaptic Weight Tensor
        self._init_gpu_tensors(sparse_weights)

        # Spikes tracking and rolling firing rate buffer
        self.history_len: int = 10
        self.step_count: int = 0
        self.last_spikes: np.ndarray = np.zeros(self.num_neurons, dtype=bool)
        self.spike_history: np.ndarray = np.zeros((self.history_len, self.num_neurons), dtype=np.float32)

        # Associative Learning & Synaptic Plasticity (KC -> MBON STDP)
        self.plastic_edge_indices: Optional[torch.Tensor] = None
        self.plastic_weights: Optional[torch.Tensor] = None
        self.baseline_plastic_weights: Optional[torch.Tensor] = None
        self.plastic_pre: Optional[torch.Tensor] = None
        self.plastic_post: Optional[torch.Tensor] = None
        self.plastic_valence_sign: Optional[torch.Tensor] = None

    @property
    def spikes(self) -> np.ndarray:
        return self.last_spikes

    def _init_gpu_tensors(self, weights: Any):
        """Compiles sparse weights and initializes membrane state vectors directly in VRAM."""
        if isinstance(weights, sp.csr_matrix):
            coo = weights.tocoo()
            indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            shape = torch.Size(coo.shape)
            self.torch_W = torch.sparse_coo_tensor(indices, values, shape, device=self.device).coalesce()
        elif isinstance(weights, torch.Tensor):
            self.torch_W = weights.to(self.device).coalesce()
        else:
            raise TypeError(f"Invalid weight matrix type: {type(weights)}")

        assert self.torch_W.shape == (self.num_neurons, self.num_neurons), (
            f"Weight shape mismatch: {self.torch_W.shape} != ({self.num_neurons}, {self.num_neurons})"
        )

        # Precompute and cache coalesced transpose matrix W^T for modern PyTorch sparse.mm
        self.torch_WT = self.torch_W.t().coalesce()

        # V: Membrane Potential Vector (mV)
        self.torch_v = torch.full((self.num_neurons,), self.v_rest, dtype=torch.float32, device=self.device)
        
        # G: Synaptic Conductance Vector
        self.torch_g = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)
        
        # Refractory Period Countdown Vector
        self.torch_refractory = torch.zeros(self.num_neurons, dtype=torch.int32, device=self.device)
        
        # Spikes Vector
        self.torch_spikes = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)

        # Transmission Delay Queue Buffer (Slots x Neurons)
        self.torch_delay_queue = torch.zeros((self.delay_slots, self.num_neurons), dtype=torch.float32, device=self.device)
        self.is_flatlined: bool = False

        logger.info(
            f"VRAM Sparse Allocation Complete: {self.torch_W._nnz():,} synaptic weights, "
            f"{self.delay_slots} conduction delay slots."
        )

    def forward_step(
        self, 
        external_current: Optional[np.ndarray] = None, 
        external_drive: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Executes one DOOMFLY biophysical integration step:
          1. Subthreshold membrane potential integration (analytic exponential solution).
          2. Synaptic conductance decay (g *= ag).
          3. Threshold evaluation (V > -45 mV).
          4. Delayed synaptic delivery via sparse matrix multiplication (W^T * S_delayed).
          5. Post-spike voltage reset (V = -52 mV) and refractory clamping.
        """
        if getattr(self, "is_flatlined", False):
            # Biological Brain Death / Flatline: 0 spikes, flatline EEG
            self.last_spikes = np.zeros(self.num_neurons, dtype=np.float32)
            return self.last_spikes

        drive_input = external_current if external_current is not None else external_drive
        with torch.no_grad():
            slot = self.cursor % self.delay_slots

            # 1. Decrement Refractory Counters
            in_ref = self.torch_refractory > 0
            self.torch_refractory[in_ref] -= 1

            # 2. External Drive Current & Deadlock Prevention Noise
            if drive_input is not None:
                assert len(drive_input) == self.num_neurons
                drive_t = torch.as_tensor(drive_input, dtype=torch.float32, device=self.device)
            else:
                drive_t = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)

            total_drive = drive_t
            if self.spontaneous_noise_std > 0:
                total_drive = total_drive + torch.randn(self.num_neurons, device=self.device, dtype=torch.float32) * self.spontaneous_noise_std

            # 3. Analytic Subthreshold Voltage Integration (DOOMFLY Equation)
            # v = -52 + (v + 52)*av + drive*(1 - av) + g*coupling
            not_ref = ~in_ref
            v_integrated = (
                self.v_rest 
                + (self.torch_v[not_ref] - self.v_rest) * self.av 
                + total_drive[not_ref] * (1.0 - self.av) 
                + self.torch_g[not_ref] * self.coupling
            )
            self.torch_v[not_ref] = v_integrated

            # 4. Conductance Decay: g *= ag (only for non-refractory neurons)
            self.torch_g[not_ref] *= self.ag

            # 5. Threshold Evaluation (Spikes Fired)
            fired = (self.torch_v >= self.v_thresh) & not_ref
            self.torch_spikes.zero_()
            self.torch_spikes[fired] = 1.0

            # Queue fired spikes into future arrival slot (1.8 ms delay)
            future_slot = (self.cursor + self.delay_slots - 1) % self.delay_slots
            self.torch_delay_queue[future_slot] = self.torch_spikes

            # 6. Deliver Synapses Arriving in Current Slot: g += W^T * S_slot
            arriving_spikes = self.torch_delay_queue[slot].unsqueeze(1)
            if torch.any(arriving_spikes > 0):
                # Modern PyTorch sparse.mm with cached coalesced W^T
                synaptic_delivery = torch.sparse.mm(self.torch_WT, arriving_spikes).squeeze(1)
                # Only deliver to non-refractory targets (Brian2 / DOOMFLY specification)
                self.torch_g[not_ref] += synaptic_delivery[not_ref]

            # Clear current delay slot
            self.torch_delay_queue[slot].zero_()

            # 7. Post-Spike Reset & Refractory Clamping
            self.torch_v[fired] = self.v_reset
            self.torch_g[fired] = 0.0
            self.torch_refractory[fired] = self.refractory_steps

            # Advance Delay Cursor
            self.cursor = (self.cursor + 1) % self.delay_slots

            # Export Spikes to CPU
            spikes_np = fired.cpu().numpy()
            self.last_spikes = spikes_np

            # Update Rolling Firing Rates
            idx = self.step_count % self.history_len
            self.spike_history[idx] = spikes_np.astype(np.float32)
            self.step_count += 1

            # Prevent VRAM fragment buildup
            if self.step_count % 500 == 0 and self.device.type == "cuda":
                torch.cuda.empty_cache()

            return spikes_np

    def forward_substeps(self, external_current: Optional[np.ndarray] = None, num_substeps: int = 4) -> np.ndarray:
        """
        Executes multiple biological LIF solver substeps sequentially on GPU
        with zero intermediate CPU memory syncs for high-throughput (60-120+ FPS) execution.
        """
        if num_substeps <= 1:
            return self.forward_step(external_current)

        if getattr(self, "is_flatlined", False):
            self.last_spikes = np.zeros(self.num_neurons, dtype=np.float32)
            return self.last_spikes

        with torch.no_grad():
            if external_current is not None:
                drive_t = torch.as_tensor(external_current, dtype=torch.float32, device=self.device)
            else:
                drive_t = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)

            fired = None
            for _ in range(num_substeps):
                slot = self.cursor % self.delay_slots
                in_ref = self.torch_refractory > 0
                self.torch_refractory[in_ref] -= 1

                total_drive = drive_t
                if self.spontaneous_noise_std > 0:
                    total_drive = total_drive + torch.randn(self.num_neurons, device=self.device, dtype=torch.float32) * self.spontaneous_noise_std

                not_ref = ~in_ref
                self.torch_v[not_ref] = (
                    self.v_rest 
                    + (self.torch_v[not_ref] - self.v_rest) * self.av 
                    + total_drive[not_ref] * (1.0 - self.av) 
                    + self.torch_g[not_ref] * self.coupling
                )
                self.torch_g[not_ref] *= self.ag

                fired = (self.torch_v >= self.v_thresh) & not_ref
                self.torch_spikes.zero_()
                self.torch_spikes[fired] = 1.0

                future_slot = (self.cursor + self.delay_slots - 1) % self.delay_slots
                self.torch_delay_queue[future_slot] = self.torch_spikes

                arriving_spikes = self.torch_delay_queue[slot].unsqueeze(1)
                if torch.any(arriving_spikes > 0):
                    synaptic_delivery = torch.sparse.mm(self.torch_WT, arriving_spikes).squeeze(1)
                    self.torch_g[not_ref] += synaptic_delivery[not_ref]

                self.torch_delay_queue[slot].zero_()
                self.torch_v[fired] = self.v_reset
                self.torch_g[fired] = 0.0
                self.torch_refractory[fired] = self.refractory_steps
                self.cursor = (self.cursor + 1) % self.delay_slots
                self.step_count += 1

            spikes_np = fired.cpu().numpy()
            self.last_spikes = spikes_np
            idx = self.step_count % self.history_len
            self.spike_history[idx] = spikes_np.astype(np.float32)

            return spikes_np

    def get_firing_rates(self) -> np.ndarray:
        """Returns smoothed firing rate vector (0.0 to 1.0) for motor decoding."""
        if getattr(self, "is_flatlined", False):
            return np.zeros(self.num_neurons, dtype=np.float32)
        return np.mean(self.spike_history, axis=0)

    def trigger_brain_death(self):
        """Simulates complete biological brain death / flatline upon character death."""
        self.is_flatlined = True
        with torch.no_grad():
            self.torch_v.fill_(self.v_reset)
            self.torch_g.zero_()
            self.torch_spikes.zero_()
            for slot in range(self.delay_slots):
                self.torch_delay_queue[slot].zero_()
            self.spike_history.fill(0.0)
            logger.warning("💀 [FLATLINE] Biological brain death triggered. 0 spikes fired.")

    def trigger_defibrillation(self):
        """Simulates defibrillation / awakening burst upon player respawn."""
        self.is_flatlined = False
        with torch.no_grad():
            self.torch_v.fill_(self.v_rest)
            # Transient defibrillation surge into central complex & descending motor tract
            self.torch_v += torch.randn(self.num_neurons, device=self.device) * 2.0
            logger.info("⚡ [DEFIBRILLATION] Resuscitation burst delivered. Brain re-awakened!")

    def init_plastic_synapses(
        self, 
        pre_indices: np.ndarray, 
        post_indices: np.ndarray,
        post_valences: Optional[np.ndarray] = None
    ):
        """
        Identifies plastic Kenyon Cell (KC) -> MBON synapses for associative learning.
        Tracks baseline efficacy to enforce biological [0.1x, 2.0x] bounding.
        Supports valence sign differentiation (Avoidance: -1.0, Approach: +1.0).
        """
        coo_indices = self.torch_W.indices() # 2 x nnz
        coo_values = self.torch_W.values()   # nnz
        
        pre_set = set(int(x) for x in pre_indices)
        post_set = set(int(x) for x in post_indices)
        
        rows = coo_indices[0].cpu().numpy()
        cols = coo_indices[1].cpu().numpy()
        
        mask = [bool((r in pre_set) and (c in post_set)) for r, c in zip(rows, cols)]
        plastic_idx = np.where(mask)[0]
        
        if len(plastic_idx) > 0:
            self.plastic_edge_indices = torch.tensor(plastic_idx, dtype=torch.long, device=self.device)
            self.plastic_weights = coo_values[self.plastic_edge_indices].clone()
            self.baseline_plastic_weights = self.plastic_weights.clone()
            self.plastic_pre = torch.tensor(rows[plastic_idx], dtype=torch.long, device=self.device)
            self.plastic_post = torch.tensor(cols[plastic_idx], dtype=torch.long, device=self.device)

            if post_valences is not None:
                val_dict = {int(idx): float(val) for idx, val in zip(post_indices, post_valences)}
                edge_valences = [val_dict.get(int(cols[i]), 1.0) for i in plastic_idx]
                self.plastic_valence_sign = torch.tensor(edge_valences, dtype=torch.float32, device=self.device)
            else:
                self.plastic_valence_sign = torch.ones(len(plastic_idx), dtype=torch.float32, device=self.device)

            logger.info(f"Initialized {len(plastic_idx):,} plastic KC -> MBON synapses for STDP learning.")
        else:
            self.plastic_edge_indices = None
            self.plastic_valence_sign = None
            logger.info("No matching KC -> MBON edges found for plastic initialization.")

    def apply_stdp_update(self, dopamine: float, eta: float = 0.002, extinction_rate: float = 0.0001):
        """
        Applies 3-factor Hebbian STDP update on Kenyon Cell -> MBON synapses:
        dW = eta * dopamine * S_pre * S_post * valence_sign
        Clamps efficacy within [0.1x, 2.0x] baseline (matching DOOMFLY specification).
        Applies passive extinction toward baseline.
        """
        if self.plastic_edge_indices is None:
            return

        has_da = abs(dopamine) >= 0.02
        has_extinction = (extinction_rate > 0 and self.step_count % 10 == 0)

        if not has_da and not has_extinction:
            return

        with torch.no_grad():
            weights_changed = False

            # Passive extinction / decay toward baseline
            if has_extinction:
                self.plastic_weights += extinction_rate * (self.baseline_plastic_weights - self.plastic_weights)
                self.plastic_weights = torch.clamp(
                    self.plastic_weights, 
                    min=0.1 * self.baseline_plastic_weights, 
                    max=2.0 * self.baseline_plastic_weights
                )
                weights_changed = True

            if has_da:
                rates = torch.as_tensor(self.get_firing_rates(), device=self.device)
                pre_activity = torch.clamp(rates[self.plastic_pre] + self.torch_spikes[self.plastic_pre], max=1.0)
                post_activity = torch.clamp(rates[self.plastic_post] + self.torch_spikes[self.plastic_post], max=1.0)
                co_activity = pre_activity * post_activity
                
                activity_signal = co_activity + 0.05 * pre_activity
                if torch.any(activity_signal > 0) or abs(dopamine) > 0.5:
                    effective_signal = torch.clamp(activity_signal, min=0.02 if abs(dopamine) > 0.5 else 0.0)
                    valence_multiplier = self.plastic_valence_sign if self.plastic_valence_sign is not None else 1.0
                    delta_w = eta * float(dopamine) * effective_signal * valence_multiplier
                    new_weights = self.plastic_weights + delta_w
                    
                    lower_bound = 0.1 * self.baseline_plastic_weights
                    upper_bound = 2.0 * self.baseline_plastic_weights
                    self.plastic_weights = torch.clamp(new_weights, min=lower_bound, max=upper_bound)
                    weights_changed = True

            if weights_changed:
                coo_indices = self.torch_W.indices()
                coo_values = self.torch_W.values().clone()
                coo_values[self.plastic_edge_indices] = self.plastic_weights
                shape = self.torch_W.shape
                self.torch_W = torch.sparse_coo_tensor(coo_indices, coo_values, shape, device=self.device).coalesce()
                self.torch_WT = self.torch_W.t().coalesce()

    def get_plasticity_telemetry(self) -> Dict[str, Any]:
        """Returns associative learning metrics matching DOOMFLY specification."""
        if self.plastic_edge_indices is None or len(self.plastic_edge_indices) == 0:
            return {"enabled": False, "num_plastic_edges": 0}
            
        ratios = (self.plastic_weights / (self.baseline_plastic_weights + 1e-6)).cpu().numpy()
        return {
            "enabled": True,
            "num_plastic_edges": len(self.plastic_edge_indices),
            "max_efficacy": float(np.max(ratios)),
            "min_efficacy": float(np.min(ratios)),
            "mean_efficacy": float(np.mean(ratios)),
            "mean_abs_change": float(np.mean(np.abs(ratios - 1.0))),
            "histogram": np.histogram(ratios, bins=10, range=(0.1, 2.0))[0].tolist()
        }
