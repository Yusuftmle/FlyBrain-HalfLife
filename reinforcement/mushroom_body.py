"""
mushroom_body.py - Drosophila Mushroom Body (Mantar Cisimciği) Learning Circuit
Models:
  - Sparse Kenyon Cell (KC) representation via APL (Anterior Paired Lateral) GABAergic inhibition (Top 5% Rule).
  - Two-pole valence output: MBON-Avoidance (Kaçınma) vs MBON-Approach (Yaklaşma).
  - 3-Factor dopamine-gated STDP: PPL1 (Pain/Damage/Stuck) vs PAM (Reward/Progress/Medkit).
  - Memory recall: Pre-emptive avoidance and appetitive approach before obstacles or traps.
  - Synaptic extinction / passive decay preventing permanent paralysis.
"""

import time
import math
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List


class MushroomBodyController:
    """
    Simulates the associative learning and behavioral valence circuit of Drosophila.
    
    Architecture:
      - 2400 Kenyon Cells (KCs): Sparse visual feature coincidence detectors (~5% active).
      - 120 MBONs:
          - Indices 0..59: MBON-Avoidance (Drives MDN backward & DNp20 turn away).
          - Indices 60..119: MBON-Approach (Drives DNpe017 forward walking).
      - PPL1 / PAM Dopamine Modulation:
          - PPL1 (Negative DA): Potentiates KC -> MBON-Avoidance, Depresses KC -> MBON-Approach.
          - PAM (Positive DA): Potentiates KC -> MBON-Approach, Depresses KC -> MBON-Avoidance.
    """

    def __init__(
        self,
        num_kc: int = 2400,
        num_mbon: int = 120,
        kc_offset: int = 9600,
        mbon_offset: int = 12000,
        sparse_ratio: float = 0.05,
        learning_rate: float = 0.015,
        extinction_rate: float = 0.0002
    ):
        self.num_kc = num_kc
        self.num_mbon = num_mbon
        self.half_mbon = num_mbon // 2  # 60 Avoidance, 60 Approach
        self.kc_offset = kc_offset
        self.mbon_offset = mbon_offset
        self.sparse_ratio = float(sparse_ratio)
        self.target_active_kc = max(1, int(round(num_kc * sparse_ratio)))  # Exactly ~120 KCs
        self.learning_rate = float(learning_rate)
        self.extinction_rate = float(extinction_rate)

        # 1. Biological Projection Matrix from Sensory Space (16-D) to KCs
        # Each KC connects randomly to 4-6 sensory input dimensions (Caron et al. Nature 2013)
        np.random.seed(42)
        self.feat_dim = 16
        self.proj_matrix = np.zeros((self.num_kc, self.feat_dim), dtype=np.float32)
        for i in range(self.num_kc):
            sample_feats = np.random.choice(self.feat_dim, size=np.random.randint(4, 7), replace=False)
            self.proj_matrix[i, sample_feats] = np.random.uniform(0.6, 1.4, size=len(sample_feats))

        # 2. KC -> MBON Synaptic Efficacies (Avoidance & Approach weights)
        # Baseline efficacy = 1.0 (clamped between 0.1 and 2.5)
        self.w_kc_avoid = np.ones((self.num_kc, self.half_mbon), dtype=np.float32) * 0.85
        self.w_kc_appr = np.ones((self.num_kc, self.half_mbon), dtype=np.float32) * 0.85
        self.w_base_avoid = self.w_kc_avoid.copy()
        self.w_base_appr = self.w_kc_appr.copy()

        # State tracking
        self.active_kc_mask = np.zeros(self.num_kc, dtype=bool)
        self.last_active_kc_indices: np.ndarray = np.array([], dtype=np.int32)
        self.current_valence: float = 0.0  # [-1.0, +1.0]
        self.last_learning_event: str = "IDLE"
        self.total_conditioned_events: int = 0
        self.escape_turn_dir: int = 1

    def extract_feature_vector(self, motion_metrics: Dict[str, Any], is_damage: bool = False, red_excess: float = 0.0) -> np.ndarray:
        """
        Builds normalized 16-dimensional sensory context vector (Conditioned Stimulus - CS).
        """
        f = np.zeros(self.feat_dim, dtype=np.float32)
        f[0] = float(motion_metrics.get("dx", 0.0)) * 2.0
        f[1] = float(motion_metrics.get("dy", 0.0)) * 2.0
        f[2] = float(motion_metrics.get("flow", 0.0)) * 10.0
        f[3] = float(motion_metrics.get("divergence", 0.0)) * 5.0
        f[4] = float(motion_metrics.get("depth_balance", 0.0)) * 3.0
        f[5] = float(motion_metrics.get("ceiling_score", 0.0))
        f[6] = float(motion_metrics.get("floor_score", 0.0))
        f[7] = float(motion_metrics.get("horizon_balance", 0.0))
        f[8] = float(motion_metrics.get("r8y_green", 0.0)) * 2.5  # Medkit / Sugar cue
        f[9] = float(motion_metrics.get("r8p_blue", 0.0)) * 2.5   # HEV cue
        f[10] = float(motion_metrics.get("ch_motion", 0.0)) * 2.0
        f[11] = float(motion_metrics.get("ch_edges", 0.0)) * 2.0
        f[12] = float(motion_metrics.get("left_eye_current", 12.0)) / 28.0
        f[13] = float(motion_metrics.get("right_eye_current", 12.0)) / 28.0
        f[14] = 1.0 if motion_metrics.get("is_obstacle_close", False) else 0.0
        f[15] = 1.0 if is_damage else float(np.clip(red_excess, 0.0, 1.0))
        return f

    def step_sensory_projection(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Computes sparse 5% Kenyon Cell excitation currents through APL feedback inhibition.
        
        Returns:
            kc_currents: Array of shape (num_kc,) with injection currents (nA).
        """
        # Linear projection into KC dendritic arbor
        kc_drive = np.dot(self.proj_matrix, feature_vec)

        # APL Global Feedback Inhibition: Top-5% winner-take-all
        # Sort activations to find threshold
        cutoff = np.partition(kc_drive, -self.target_active_kc)[-self.target_active_kc]
        active_mask = (kc_drive >= cutoff)

        # Enforce exact top-5% count
        active_indices = np.where(active_mask)[0][:self.target_active_kc]
        self.active_kc_mask = np.zeros(self.num_kc, dtype=bool)
        self.active_kc_mask[active_indices] = True
        self.last_active_kc_indices = active_indices

        # Active KCs receive strong depolarizing current (32 nA > 15 nA threshold) to fire bursts
        kc_currents = np.zeros(self.num_kc, dtype=np.float32)
        kc_currents[active_indices] = 32.0
        return kc_currents

    def compute_valence_and_recall(self, firing_rates: np.ndarray) -> Dict[str, Any]:
        """
        Reads MBON-Avoidance and MBON-Approach firing rates from LIF engine and computes net valence.
        
        Valence V_MB in [-1.0, +1.0]:
          V_MB < -0.05: Threat / Aversive memory recall -> Back up (MDN) and turn away (DNp20).
          V_MB > +0.05: Safety / Appetitive memory recall -> Confident forward exploration (DNpe017).
        """
        if len(firing_rates) <= self.mbon_offset:
            return {"valence": 0.0, "is_avoidance_recall": False, "is_approach_recall": False}

        mbon_rates = firing_rates[self.mbon_offset : self.mbon_offset + self.num_mbon]
        if len(mbon_rates) < self.num_mbon:
            return {"valence": 0.0, "is_avoidance_recall": False, "is_approach_recall": False}

        # Subpopulations
        rate_avoid = float(np.mean(mbon_rates[:self.half_mbon]))
        rate_appr = float(np.mean(mbon_rates[self.half_mbon:]))

        # Net Valence
        # Positive = Appetitive approach, Negative = Aversive avoidance
        diff = rate_appr - rate_avoid
        self.current_valence = float(np.clip(diff * 3.5, -1.0, 1.0))

        is_avoid = bool(self.current_valence < -0.06)
        is_appr = bool(self.current_valence > 0.06)

        # Aversive behavioral modulation
        # If danger is recalled: compute motor biases
        mdn_override_current = 0.0
        turn_override_bias = 0.0
        forward_suppression = 1.0

        if is_avoid:
            severity = abs(self.current_valence)
            # MDN burst override to step backward
            mdn_override_current = severity * 35.0
            # Steer away from the danger direction
            turn_override_bias = self.escape_turn_dir * severity * 60.0
            # Suppress forward walking into known hazard
            forward_suppression = max(0.0, 1.0 - severity * 1.5)
            self.last_learning_event = "RECALL_AVOIDANCE"
        elif is_appr:
            self.last_learning_event = "RECALL_APPROACH"

        return {
            "valence": self.current_valence,
            "rate_avoid": rate_avoid,
            "rate_appr": rate_appr,
            "is_avoidance_recall": is_avoid,
            "is_approach_recall": is_appr,
            "mdn_stim": mdn_override_current,
            "steer_bias": turn_override_bias,
            "forward_suppression": forward_suppression
        }

    def apply_dopamine_plasticity(self, dopamine_level: float, escape_dir: int = 1) -> None:
        """
        Applies 3-Factor Hebbian STDP to KC -> MBON synapses modulated by PPL1/PAM dopamine.
        
        Args:
            dopamine_level: Negative for punishment (PPL1), Positive for reward (PAM).
            escape_dir: Current escape turn direction (+1 right, -1 left).
        """
        if len(self.last_active_kc_indices) == 0:
            return

        self.escape_turn_dir = escape_dir

        # 1. PPL1 Punishment Conditioning (Damage / Pain / Wall-Stuck)
        if dopamine_level < -0.15:
            penalty_mag = abs(dopamine_level)
            delta = self.learning_rate * penalty_mag

            # Strengthen active KC -> MBON-Avoidance synapses
            self.w_kc_avoid[self.last_active_kc_indices, :] += delta
            # Weaken active KC -> MBON-Approach synapses
            self.w_kc_appr[self.last_active_kc_indices, :] -= delta * 0.8

            self.last_learning_event = "PPL1_PUNISHMENT"
            self.total_conditioned_events += 1

        # 2. PAM Reward Conditioning (Progress / Medkit / Target Kill)
        elif dopamine_level > 0.15:
            reward_mag = dopamine_level
            delta = self.learning_rate * reward_mag

            # Strengthen active KC -> MBON-Approach synapses
            self.w_kc_appr[self.last_active_kc_indices, :] += delta
            # Weaken active KC -> MBON-Avoidance synapses
            self.w_kc_avoid[self.last_active_kc_indices, :] -= delta * 0.8

            self.last_learning_event = "PAM_REWARD"
            self.total_conditioned_events += 1

        # 3. Enforce Biological Bounds [0.1, 2.5] of baseline
        self.w_kc_avoid = np.clip(self.w_kc_avoid, 0.1 * self.w_base_avoid, 2.5 * self.w_base_avoid)
        self.w_kc_appr = np.clip(self.w_kc_appr, 0.1 * self.w_base_appr, 2.5 * self.w_base_appr)

        # 4. Passive Memory Extinction / Decay (Prevents permanent fear paralysis)
        self.w_kc_avoid += self.extinction_rate * (self.w_base_avoid - self.w_kc_avoid)
        self.w_kc_appr += self.extinction_rate * (self.w_base_appr - self.w_kc_appr)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns associative memory telemetry metrics."""
        active_kc_count = int(np.sum(self.active_kc_mask))
        active_pct = (active_kc_count / max(1, self.num_kc)) * 100.0
        mean_avoid_eff = float(np.mean(self.w_kc_avoid / (self.w_base_avoid + 1e-6)))
        mean_appr_eff = float(np.mean(self.w_kc_appr / (self.w_base_appr + 1e-6)))

        return {
            "valence": round(self.current_valence, 3),
            "active_kc_count": active_kc_count,
            "active_kc_pct": round(active_pct, 1),
            "mean_avoid_efficacy": round(mean_avoid_eff, 3),
            "mean_appr_efficacy": round(mean_appr_eff, 3),
            "total_conditioned_events": self.total_conditioned_events,
            "last_event": self.last_learning_event
        }
