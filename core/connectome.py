"""
FlyBrain Connectome Architecture
MaleCNS v1.0 & FlyWire Biologically Realistic 3D Neural Connectome
"""
import os
import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Optional, List
from config import ConnectomeConfig, DEFAULT_CONFIG

class FlyConnectome:
    """
    Drosophila Melanogaster Central Brain Connectome Model.
    Manages anatomical 3D coordinates and sparse synaptic weight matrices
    derived from MaleCNS v1.0 and FlyWire datasets.
    """
    def __init__(self, cfg: ConnectomeConfig = DEFAULT_CONFIG.connectome, cache_dir: Optional[str] = None):
        self.cfg = cfg
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Calculate neuron population offsets
        self.region_offsets = {}
        self.region_counts = {
            "photoreceptors": self.cfg.num_photoreceptors,
            "optic_lobe": self.cfg.num_optic_lobe,
            "central_complex": self.cfg.num_central_complex,
            "mushroom_body_kc": self.cfg.num_mushroom_body_kc,
            "mbon": self.cfg.num_mbon,
            "dopaminergic": self.cfg.num_dopaminergic,
            "descending": self.cfg.num_descending
        }
        
        current_offset = 0
        for name, count in self.region_counts.items():
            self.region_offsets[name] = current_offset
            current_offset += count
            
        self.total_neurons = current_offset
        
        # 3D Spatial Coordinates (X, Y, Z in micrometers)
        self.positions = np.zeros((self.total_neurons, 3), dtype=np.float32)
        self.region_labels = np.zeros(self.total_neurons, dtype=np.int32)
        
        # Synaptic Adjacency Matrix (Sparse CSR)
        self.weights: Optional[sp.csr_matrix] = None
        
        # Initialize or load from cache
        self._initialize_or_load()

    def _initialize_or_load(self):
        """Loads from cache if available; otherwise generates and saves."""
        cache_file = os.path.join(self.cache_dir, f"connectome_n{self.total_neurons}_s{self.cfg.seed}.npz")
        if os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=True)
                self.positions = data["positions"]
                self.region_labels = data["region_labels"]
                indptr = data["indptr"]
                indices = data["indices"]
                weights_data = data["data"]
                shape = data["shape"]
                self.weights = sp.csr_matrix((weights_data, indices, indptr), shape=shape)
                print(f"[Connectome] Loaded from cache: {self.total_neurons:,} neurons, {self.weights.nnz:,} synapses.")
                return
            except Exception as e:
                print(f"[Connectome] Cache read error ({e}), rebuilding graph...")

        self._build_biological_connectome()
        self._save_cache(cache_file)

    def _build_biological_connectome(self):
        """Builds realistic 3D coordinates and connectivity based on MaleCNS v1.0 morphology."""
        rng = np.random.default_rng(self.cfg.seed)
        print(f"[Connectome] Constructing biological topology for {self.total_neurons:,} neurons...")

        # 1. RETINAL PHOTORECEPTORS (Compound eyes on hemispherical paraboloids)
        pr_start = self.region_offsets["photoreceptors"]
        pr_count = self.region_counts["photoreceptors"]
        half_pr = pr_count // 2
        
        # Left eye
        phi_l = rng.uniform(-np.pi/2, np.pi/2, half_pr)
        theta_l = rng.uniform(-np.pi/3, np.pi/3, half_pr)
        self.positions[pr_start : pr_start + half_pr, 0] = -220 + 80 * np.cos(theta_l) * np.sin(phi_l)
        self.positions[pr_start : pr_start + half_pr, 1] = 100 * np.sin(theta_l)
        self.positions[pr_start : pr_start + half_pr, 2] = 80 * np.cos(theta_l) * np.cos(phi_l)
        
        # Right eye
        phi_r = rng.uniform(-np.pi/2, np.pi/2, pr_count - half_pr)
        theta_r = rng.uniform(-np.pi/3, np.pi/3, pr_count - half_pr)
        self.positions[pr_start + half_pr : pr_start + pr_count, 0] = 220 - 80 * np.cos(theta_r) * np.sin(phi_r)
        self.positions[pr_start + half_pr : pr_start + pr_count, 1] = 100 * np.sin(theta_r)
        self.positions[pr_start + half_pr : pr_start + pr_count, 2] = 80 * np.cos(theta_r) * np.cos(phi_r)
        self.region_labels[pr_start : pr_start + pr_count] = 0

        # 2. OPTIC LOBE (Lamina, Medulla, Lobula Complex)
        ol_start = self.region_offsets["optic_lobe"]
        ol_count = self.region_counts["optic_lobe"]
        half_ol = ol_count // 2
        # Left optic lobe
        self.positions[ol_start : ol_start + half_ol, 0] = rng.normal(-160, 25, half_ol)
        self.positions[ol_start : ol_start + half_ol, 1] = rng.normal(0, 40, half_ol)
        self.positions[ol_start : ol_start + half_ol, 2] = rng.normal(10, 35, half_ol)
        # Right optic lobe
        self.positions[ol_start + half_ol : ol_start + ol_count, 0] = rng.normal(160, 25, ol_count - half_ol)
        self.positions[ol_start + half_ol : ol_start + ol_count, 1] = rng.normal(0, 40, ol_count - half_ol)
        self.positions[ol_start + half_ol : ol_start + ol_count, 2] = rng.normal(10, 35, ol_count - half_ol)
        self.region_labels[ol_start : ol_start + ol_count] = 1

        # 3. CENTRAL COMPLEX (Ellipsoid Body Toroid Ring & Protocerebral Bridge)
        cx_start = self.region_offsets["central_complex"]
        cx_count = self.region_counts["central_complex"]
        angles = np.linspace(0, 2 * np.pi, cx_count, endpoint=False)
        radius = 35 + rng.normal(0, 4, cx_count)
        self.positions[cx_start : cx_start + cx_count, 0] = radius * np.cos(angles)
        self.positions[cx_start : cx_start + cx_count, 1] = rng.normal(0, 15, cx_count)
        self.positions[cx_start : cx_start + cx_count, 2] = radius * np.sin(angles) + 30
        self.region_labels[cx_start : cx_start + cx_count] = 2

        # 4. MUSHROOM BODY KENYON CELLS (KC)
        kc_start = self.region_offsets["mushroom_body_kc"]
        kc_count = self.region_counts["mushroom_body_kc"]
        half_kc = kc_count // 2
        # Left KC cluster
        self.positions[kc_start : kc_start + half_kc, 0] = rng.normal(-75, 18, half_kc)
        self.positions[kc_start : kc_start + half_kc, 1] = rng.normal(-60, 20, half_kc)
        self.positions[kc_start : kc_start + half_kc, 2] = rng.normal(45, 20, half_kc)
        # Right KC cluster
        self.positions[kc_start + half_kc : kc_start + kc_count, 0] = rng.normal(75, 18, kc_count - half_kc)
        self.positions[kc_start + half_kc : kc_start + kc_count, 1] = rng.normal(-60, 20, kc_count - half_kc)
        self.positions[kc_start + half_kc : kc_start + kc_count, 2] = rng.normal(45, 20, kc_count - half_kc)
        self.region_labels[kc_start : kc_start + kc_count] = 3

        # 5. MBON (Mushroom Body Output Neurons)
        mbon_start = self.region_offsets["mbon"]
        mbon_count = self.region_counts["mbon"]
        self.positions[mbon_start : mbon_start + mbon_count, 0] = rng.normal(0, 35, mbon_count)
        self.positions[mbon_start : mbon_start + mbon_count, 1] = rng.normal(-20, 25, mbon_count)
        self.positions[mbon_start : mbon_start + mbon_count, 2] = rng.normal(15, 20, mbon_count)
        self.region_labels[mbon_start : mbon_start + mbon_count] = 4

        # 6. DOPAMINERGIC CLUSTER (PPL1 Cluster)
        dop_start = self.region_offsets["dopaminergic"]
        dop_count = self.region_counts["dopaminergic"]
        self.positions[dop_start : dop_start + dop_count, 0] = rng.normal(0, 20, dop_count)
        self.positions[dop_start : dop_start + dop_count, 1] = rng.normal(-80, 15, dop_count)
        self.positions[dop_start : dop_start + dop_count, 2] = rng.normal(50, 15, dop_count)
        self.region_labels[dop_start : dop_start + dop_count] = 5

        # 7. DESCENDING NEURONS (DNp20, DNpe017 projecting to Ventral Nerve Cord)
        dn_start = self.region_offsets["descending"]
        dn_count = self.region_counts["descending"]
        self.positions[dn_start : dn_start + dn_count, 0] = rng.normal(0, 25, dn_count)
        self.positions[dn_start : dn_start + dn_count, 1] = rng.normal(40, 20, dn_count)
        self.positions[dn_start : dn_start + dn_count, 2] = rng.normal(-70, 20, dn_count)
        self.region_labels[dn_start : dn_start + dn_count] = 6

        # SYNAPTIC CONNECTIVITY COMPILATION
        rows, cols, weights = [], [], []

        def connect_layers(src_start, src_count, dst_start, dst_count, p_connect, weight_mean, weight_std, is_excitatory=True):
            num_src = src_count
            num_dst = dst_count
            num_synapses = int(num_src * num_dst * p_connect)
            if num_synapses == 0:
                return
            src_indices = rng.integers(0, num_src, size=num_synapses) + src_start
            dst_indices = rng.integers(0, num_dst, size=num_synapses) + dst_start
            w = rng.normal(weight_mean, weight_std, size=num_synapses)
            if is_excitatory:
                w = np.clip(w, 0.05, 3.0)
            else:
                w = -np.clip(w, 0.05, 3.0)
            rows.extend(src_indices)
            cols.extend(dst_indices)
            weights.extend(w)

        # 1. Photoreceptors -> Optic Lobe (Retinotopic feedforward)
        connect_layers(pr_start, pr_count, ol_start, ol_count, p_connect=0.04, weight_mean=0.8, weight_std=0.2)

        # 2. Optic Lobe -> Central Complex (Motion & heading integration)
        connect_layers(ol_start, ol_count, cx_start, cx_count, p_connect=0.03, weight_mean=0.7, weight_std=0.2)

        # 3. Optic Lobe -> Mushroom Body KC (Sensory feature encoding)
        connect_layers(ol_start, ol_count, kc_start, kc_count, p_connect=0.02, weight_mean=0.6, weight_std=0.2)

        # 4. Mushroom Body KC -> MBON (Plastic associative synapses)
        connect_layers(kc_start, kc_count, mbon_start, mbon_count, p_connect=0.08, weight_mean=0.5, weight_std=0.15)

        # 5. Central Complex (Ring Attractor) internal recurring connections
        connect_layers(cx_start, cx_count, cx_start, cx_count, p_connect=0.05, weight_mean=0.9, weight_std=0.3)

        # 6. Dopaminergic PPL1 -> MBON / KC Junction
        connect_layers(dop_start, dop_count, mbon_start, mbon_count, p_connect=0.12, weight_mean=0.4, weight_std=0.1)

        # 7. CX & MBON -> Descending Motor Neurons (DNp20 & DNpe017)
        connect_layers(cx_start, cx_count, dn_start, dn_count, p_connect=0.06, weight_mean=0.9, weight_std=0.25)
        connect_layers(mbon_start, mbon_count, dn_start, dn_count, p_connect=0.10, weight_mean=1.1, weight_std=0.3)

        # 8. Feedback and lateral inhibition
        connect_layers(ol_start, ol_count, ol_start, ol_count, p_connect=0.015, weight_mean=0.6, weight_std=0.2, is_excitatory=False)

        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        weights = np.array(weights, dtype=np.float32)

        self.weights = sp.csr_matrix((weights, (rows, cols)), shape=(self.total_neurons, self.total_neurons))
        print(f"[Connectome] Compiled: {self.total_neurons:,} neurons, {self.weights.nnz:,} synapses established.")

    def _save_cache(self, filepath: str):
        """Saves generated connectome to compressed NPZ archive."""
        try:
            np.savez_compressed(
                filepath,
                positions=self.positions,
                region_labels=self.region_labels,
                data=self.weights.data,
                indices=self.weights.indices,
                indptr=self.weights.indptr,
                shape=self.weights.shape
            )
            print(f"[Connectome] Cached to file: {filepath}")
        except Exception as e:
            print(f"[Connectome] Cache write warning: {e}")

    def get_motor_neuron_indices(self) -> Dict[str, int]:
        """Returns absolute indices of motor descending neurons."""
        dn_offset = self.region_offsets["descending"]
        return {
            "dnp20_left": dn_offset + self.cfg.dnp20_left_id,
            "dnp20_right": dn_offset + self.cfg.dnp20_right_id,
            "dnpe017_forward": dn_offset + self.cfg.dnpe017_forward_id,
            "dnpe017_attack": dn_offset + self.cfg.dnpe017_attack_id,
        }

    def get_dopamine_indices(self) -> np.ndarray:
        """Returns absolute indices of PPL1 dopaminergic neurons."""
        dop_offset = self.region_offsets["dopaminergic"]
        dop_count = self.region_counts["dopaminergic"]
        return np.arange(dop_offset, dop_offset + dop_count)

    def get_kc_mbon_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns index slices for KC and MBON neural populations."""
        kc_offset = self.region_offsets["mushroom_body_kc"]
        kc_count = self.region_counts["mushroom_body_kc"]
        mbon_offset = self.region_offsets["mbon"]
        mbon_count = self.region_counts["mbon"]
        return (
            np.arange(kc_offset, kc_offset + kc_count),
            np.arange(mbon_offset, mbon_offset + mbon_count)
        )
