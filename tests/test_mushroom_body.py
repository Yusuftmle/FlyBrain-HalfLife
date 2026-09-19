"""
test_mushroom_body.py - Unit tests for Mushroom Body (Mantar Cisimciği) Learning Circuit
Tests:
  1. 16-D Sensory Scene Feature Extraction
  2. Sparse 5% Kenyon Cell Representation via APL Inhibition
  3. PPL1 Aversive Punishment Conditioning (Avoidance strengthening)
  4. PAM Appetitive Reward Conditioning (Approach strengthening)
  5. Memory Recall and Closed-Loop Motor Biasing (MDN & Steering)
  6. Passive Synaptic Extinction / Decay
  7. Full LIF Engine & Connectome Plastic Integration
"""
import unittest
import numpy as np
import scipy.sparse as sp
import torch

from reinforcement.mushroom_body import MushroomBodyController
from lif_engine import PyTorchLIFEngine


class TestMushroomBody(unittest.TestCase):

    def setUp(self):
        self.num_kc = 2400
        self.num_mbon = 120
        self.mb = MushroomBodyController(
            num_kc=self.num_kc,
            num_mbon=self.num_mbon,
            kc_offset=9600,
            mbon_offset=12000,
            sparse_ratio=0.05,
            learning_rate=0.03,
            extinction_rate=0.001
        )

    def test_sensory_feature_extraction(self):
        """Verifies that 16-D sensory context vectors are properly built."""
        motion_metrics = {
            "dx": 0.05,
            "dy": -0.02,
            "flow": 0.002,
            "divergence": 0.01,
            "depth_balance": 0.35,
            "ceiling_score": 0.15,
            "floor_score": 0.25,
            "horizon_balance": -0.10,
            "r8y_green": 1.2,
            "r8p_blue": 0.5,
            "ch_motion": 0.04,
            "ch_edges": 0.08,
            "left_eye_current": 14.0,
            "right_eye_current": 10.0,
            "is_obstacle_close": True
        }
        feat = self.mb.extract_feature_vector(motion_metrics, is_damage=True, red_excess=0.8)
        self.assertEqual(len(feat), 16)
        self.assertAlmostEqual(feat[14], 1.0)  # Obstacle present
        self.assertAlmostEqual(feat[15], 1.0)  # Damage active
        self.assertGreater(feat[8], 0.0)      # Green medkit cue

    def test_sparse_5_percent_kc_rule(self):
        """Validates that APL feedback inhibition enforces exactly 5% sparse KC activation."""
        feat = np.random.uniform(0.1, 1.0, size=16).astype(np.float32)
        kc_stim = self.mb.step_sensory_projection(feat)

        self.assertEqual(len(kc_stim), self.num_kc)
        # Exactly 5% of 2400 = 120 KCs
        expected_active = int(round(self.num_kc * 0.05))
        active_indices = np.where(kc_stim > 0)[0]
        self.assertEqual(len(active_indices), expected_active)
        self.assertTrue(np.all(kc_stim[active_indices] == 32.0))
        self.assertEqual(np.sum(kc_stim == 0.0), self.num_kc - expected_active)

        diag = self.mb.get_diagnostics()
        self.assertEqual(diag["active_kc_count"], 120)
        self.assertAlmostEqual(diag["active_kc_pct"], 5.0, places=1)

    def test_ppl1_aversive_conditioning(self):
        """Validates that damage/pain (PPL1) strengthens KC -> MBON-Avoidance synapses."""
        feat = np.zeros(16, dtype=np.float32)
        feat[15] = 1.0  # Damage cue
        self.mb.step_sensory_projection(feat)

        baseline_avoid = np.mean(self.mb.w_kc_avoid)
        baseline_appr = np.mean(self.mb.w_kc_appr)

        # Apply negative dopamine punishment (PPL1)
        for _ in range(5):
            self.mb.apply_dopamine_plasticity(dopamine_level=-1.5, escape_dir=-1)

        new_avoid = np.mean(self.mb.w_kc_avoid)
        new_appr = np.mean(self.mb.w_kc_appr)

        self.assertGreater(new_avoid, baseline_avoid, "Avoidance synapses must potentiate on punishment")
        self.assertLess(new_appr, baseline_appr, "Approach synapses must depress on punishment")

    def test_pam_reward_conditioning(self):
        """Validates that reward/sugar/health (PAM) strengthens KC -> MBON-Approach synapses."""
        feat = np.zeros(16, dtype=np.float32)
        feat[8] = 2.0  # Health pack / sugar cue
        self.mb.step_sensory_projection(feat)

        baseline_avoid = np.mean(self.mb.w_kc_avoid)
        baseline_appr = np.mean(self.mb.w_kc_appr)

        # Apply positive dopamine reward (PAM)
        for _ in range(5):
            self.mb.apply_dopamine_plasticity(dopamine_level=1.5, escape_dir=1)

        new_avoid = np.mean(self.mb.w_kc_avoid)
        new_appr = np.mean(self.mb.w_kc_appr)

        self.assertGreater(new_appr, baseline_appr, "Approach synapses must potentiate on reward")
        self.assertLess(new_avoid, baseline_avoid, "Avoidance synapses must depress on reward")

    def test_memory_recall_and_motor_modulation(self):
        """Validates that recalled fear drives MDN backward step, turn bias, and forward suppression."""
        total_neurons = 12000 + self.num_mbon
        firing_rates = np.zeros(total_neurons, dtype=np.float32)

        # Simulate conditioned threat: MBON-Avoidance (indices 12000..12059) fire strongly
        firing_rates[12000 : 12060] = 0.45
        firing_rates[12060 : 12120] = 0.05

        self.mb.escape_turn_dir = 1
        recall = self.mb.compute_valence_and_recall(firing_rates)

        self.assertLess(recall["valence"], -0.06, "Threat recall must produce negative valence")
        self.assertTrue(recall["is_avoidance_recall"])
        self.assertGreater(recall["mdn_stim"], 10.0, "Threat recall must stimulate MDN backward walking")
        self.assertGreater(recall["steer_bias"], 0.0, "Threat recall must trigger escape steer bias")
        self.assertLess(recall["forward_suppression"], 1.0, "Threat recall must suppress forward drive")

    def test_passive_extinction(self):
        """Validates that unused fear associations decay back to baseline over time."""
        feat = np.ones(16, dtype=np.float32)
        self.mb.step_sensory_projection(feat)

        # Severe shock
        self.mb.apply_dopamine_plasticity(dopamine_level=-2.0)
        elevated_avoid = np.mean(self.mb.w_kc_avoid)

        # Decay through neutral steps
        for _ in range(100):
            self.mb.apply_dopamine_plasticity(dopamine_level=0.0)

        decayed_avoid = np.mean(self.mb.w_kc_avoid)
        self.assertLess(decayed_avoid, elevated_avoid, "Passive extinction must relax weights toward baseline")

    def test_lif_engine_plastic_valence_integration(self):
        """Validates that PyTorchLIFEngine updates plastic synapses according to post-synaptic valence signs."""
        num_neurons = 300
        # 100 KCs (0..99) -> 50 Avoidance MBONs (100..149) and 50 Approach MBONs (150..199)
        rows, cols, data = [], [], []
        for pre in range(100):
            for post in range(100, 200):
                rows.append(pre)
                cols.append(post)
                data.append(0.6)
        csr = sp.csr_matrix((data, (rows, cols)), shape=(num_neurons, num_neurons), dtype=np.float32)

        engine = PyTorchLIFEngine(sparse_weights=csr, num_neurons=num_neurons, device="cpu")
        kc_idx = np.arange(0, 100)
        mbon_idx = np.arange(100, 200)

        # First half = Avoidance (-1.0), Second half = Approach (+1.0)
        valences = np.ones(100, dtype=np.float32)
        valences[:50] = -1.0

        engine.init_plastic_synapses(kc_idx, mbon_idx, post_valences=valences)
        self.assertIsNotNone(engine.plastic_valence_sign)

        # Drive all neurons to spike
        drive = np.zeros(num_neurons, dtype=np.float32)
        drive[:200] = 35.0
        for _ in range(4):
            engine.forward_step(drive)

        # Apply negative dopamine punishment (PPL1)
        for _ in range(15):
            engine.apply_stdp_update(dopamine=-1.5, eta=0.02)

        w_updated = engine.plastic_weights.cpu().numpy()
        w_base = engine.baseline_plastic_weights.cpu().numpy()
        cols_arr = engine.plastic_post.cpu().numpy()

        # Avoidance edges (post < 150) should be potentiated (ratio > 1.0)
        avoid_mask = (cols_arr < 150)
        # Approach edges (post >= 150) should be depressed (ratio < 1.0)
        appr_mask = (cols_arr >= 150)

        mean_avoid_ratio = np.mean(w_updated[avoid_mask] / w_base[avoid_mask])
        mean_appr_ratio = np.mean(w_updated[appr_mask] / w_base[appr_mask])

        self.assertGreater(mean_avoid_ratio, 1.0, "Negative dopamine must potentiate MBON-Avoidance edges")
        self.assertLess(mean_appr_ratio, 1.0, "Negative dopamine must depress MBON-Approach edges")


if __name__ == "__main__":
    unittest.main()
