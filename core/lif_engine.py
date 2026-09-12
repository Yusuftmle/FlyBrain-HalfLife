"""
LIF Engine (Leaky Integrate-and-Fire Neural Network Simulator)
High-Performance Spiking Neural Simulation with Sparse Matrix Propagation
Includes Deadlock Prevention via Spontaneous Background Noise
"""
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Dict
from config import SimulationConfig, DEFAULT_CONFIG
from core.connectome import FlyConnectome

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class LIFEngine:
    """
    Biological Spiking Neural Network Simulator over Drosophila Connectome.
    Vectorized Leaky Integrate-and-Fire differential equation solver.
    Prevents silent deadlock using synthetic Poisson/Gaussian spontaneous noise.
    """
    def __init__(self, connectome: FlyConnectome, cfg: SimulationConfig = DEFAULT_CONFIG.sim):
        self.connectome = connectome
        self.cfg = cfg
        self.num_neurons = connectome.total_neurons
        
        # Mathematical constants
        self.decay_factor = float(np.exp(-self.cfg.dt / self.cfg.tau_m))
        
        # Neuron State Vectors (NumPy)
        self.v = np.full(self.num_neurons, self.cfg.v_rest, dtype=np.float32)
        self.spikes = np.zeros(self.num_neurons, dtype=bool)
        self.refractory = np.zeros(self.num_neurons, dtype=np.int32)
        self.synaptic_current = np.zeros(self.num_neurons, dtype=np.float32)
        
        # Rolling spike history buffer for rate decoding
        self.spike_history_len = 10
        self.spike_history = np.zeros((self.spike_history_len, self.num_neurons), dtype=np.float32)
        self.step_counter = 0
        
        # PyTorch Tensor Acceleration (GPU/CPU)
        self.use_torch = TORCH_AVAILABLE and self.cfg.use_pytorch
        self.device = self.cfg.device if self.use_torch else "cpu"
        
        if self.use_torch:
            self._init_pytorch_tensors()
            print(f"[LIF Engine] PyTorch Sparse Acceleration enabled. Device: {self.device.upper()}")
        else:
            print("[LIF Engine] NumPy / SciPy Sparse Engine active (Ultra-optimized CPU).")

    def _init_pytorch_tensors(self):
        """Converts SciPy CSR matrix into PyTorch Sparse COO tensor."""
        coo = self.connectome.weights.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(coo.shape)
        
        self.torch_weights = torch.sparse_coo_tensor(indices, values, shape).to(self.device).coalesce()
        self.torch_v = torch.full((self.num_neurons,), self.cfg.v_rest, dtype=torch.float32, device=self.device)
        self.torch_spikes = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
        self.torch_refractory = torch.zeros(self.num_neurons, dtype=torch.int32, device=self.device)

    def step(self, external_current: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advances the network state by one discrete time step (dt).
        Input: external_current (Photoreceptors / Dopamine injection)
        Output: spikes (Boolean array of firing neurons)
        """
        if self.use_torch:
            return self._step_pytorch(external_current)
        else:
            return self._step_numpy(external_current)

    def _step_numpy(self, external_current: Optional[np.ndarray]) -> np.ndarray:
        # 1. Decrement refractory period counters
        in_refractory = self.refractory > 0
        self.refractory[in_refractory] -= 1
        
        # 2. Compute Synaptic Input Current: I_syn = W^T * Spikes(t-1)
        active_indices = np.where(self.spikes)[0]
        if len(active_indices) > 0:
            sub_mat = self.connectome.weights[active_indices, :]
            new_syn = np.array(sub_mat.sum(axis=0)).flatten()
            self.synaptic_current = self.synaptic_current * self.cfg.synaptic_decay + new_syn
        else:
            self.synaptic_current *= self.cfg.synaptic_decay
            
        # 3. External Current + Biological Spontaneous Background Noise (Deadlock Prevention)
        i_ext = 0.0 if external_current is None else external_current
        if self.cfg.spontaneous_noise_std > 0:
            spontaneous_noise = np.random.normal(
                self.cfg.spontaneous_bias, 
                self.cfg.spontaneous_noise_std, 
                self.num_neurons
            ).astype(np.float32)
        else:
            spontaneous_noise = 0.0

        # 4. Membrane Potential Leakage & Integration
        not_refractory = ~in_refractory
        v_diff = (self.v[not_refractory] - self.cfg.v_rest) * self.decay_factor
        
        total_input = self.synaptic_current[not_refractory] + spontaneous_noise[not_refractory]
        if isinstance(i_ext, np.ndarray):
            total_input += i_ext[not_refractory]
        else:
            total_input += i_ext

        self.v[not_refractory] = self.cfg.v_rest + v_diff + total_input

        # 5. Threshold Evaluation & Spike Generation
        fired = (self.v >= self.cfg.v_thresh) & not_refractory
        self.spikes[:] = False
        self.spikes[fired] = True

        # 6. Post-Spike Reset and Refractory Clamping
        self.v[fired] = self.cfg.v_reset
        self.refractory[fired] = self.cfg.refractory_period

        # 7. Update Rolling History
        idx = self.step_counter % self.spike_history_len
        self.spike_history[idx] = self.spikes.astype(np.float32)
        self.step_counter += 1

        return self.spikes

    def _step_pytorch(self, external_current: Optional[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            in_ref = self.torch_refractory > 0
            self.torch_refractory[in_ref] -= 1

            active_spikes = self.torch_spikes.float().unsqueeze(1)
            syn_in = torch.sparse.mm(self.torch_weights.t(), active_spikes).squeeze(1)

            if external_current is not None:
                ext_t = torch.as_tensor(external_current, dtype=torch.float32, device=self.device)
            else:
                ext_t = 0.0

            if self.cfg.spontaneous_noise_std > 0:
                spontaneous_noise = torch.randn(self.num_neurons, device=self.device) * self.cfg.spontaneous_noise_std + self.cfg.spontaneous_bias
            else:
                spontaneous_noise = 0.0

            not_ref = ~in_ref
            total_in = syn_in[not_ref] + spontaneous_noise[not_ref]
            if isinstance(ext_t, torch.Tensor):
                total_in += ext_t[not_ref]
            else:
                total_in += ext_t

            self.torch_v[not_ref] = self.cfg.v_rest + (self.torch_v[not_ref] - self.cfg.v_rest) * self.decay_factor + total_in

            fired = (self.torch_v >= self.cfg.v_thresh) & not_ref
            self.torch_spikes.zero_()
            self.torch_spikes[fired] = True

            self.torch_v[fired] = self.cfg.v_reset
            self.torch_refractory[fired] = self.cfg.refractory_period

            self.spikes = self.torch_spikes.cpu().numpy()
            self.v = self.torch_v.cpu().numpy()

            idx = self.step_counter % self.spike_history_len
            self.spike_history[idx] = self.spikes.astype(np.float32)
            self.step_counter += 1

            return self.spikes

    def get_firing_rates(self) -> np.ndarray:
        """Returns sliding-window firing rates (0.0 to 1.0) for all neurons."""
        return np.mean(self.spike_history, axis=0)

    def get_region_activity(self, region_name: str) -> float:
        """Returns mean spike rate in a designated anatomical cluster."""
        offset = self.connectome.region_offsets[region_name]
        count = self.connectome.region_counts[region_name]
        return float(np.mean(self.spikes[offset : offset + count]))
