"""
test_dopamine_streak.py - Unit tests for:
  1. Clean forward exploration streak dopamine reinforcement & LB3c sugar stimulation.
  2. Mushroom Body appetitive memory STDP plasticity under streak rewards.
"""
import unittest
import numpy as np

from dopamine import DopamineController
from reinforcement.mushroom_body import MushroomBodyController
from config import DEFAULT_CONFIG


class MockConnectome:
    def __init__(self):
        self.total_neurons = 1000
    def get_dopamine_indices(self):
        return np.arange(10, 20)
    def get_kc_mbon_indices(self):
        return np.arange(50, 100), np.arange(100, 120)


class MockLIFEngine:
    def __init__(self):
        self.synaptic_weights = None
        self.spikes = np.zeros(1000, dtype=bool)


class TestDopamineStreak(unittest.TestCase):

    def setUp(self):
        self.connectome = MockConnectome()
        self.lif = MockLIFEngine()
        self.da = DopamineController(self.connectome, self.lif, cfg=DEFAULT_CONFIG.rl)
        self.mb = MushroomBodyController(num_kc=200, num_mbon=20, mbon_offset=0)

    def test_streak_dopamine_elevation(self):
        """Validates that a clean movement streak elevates dopamine and records reward."""
        init_da = self.da.dopamine_level
        self.da.register_event("streak", magnitude=1.0)
        self.assertGreater(self.da.dopamine_level, init_da)
        self.assertGreater(self.da.total_rewards, 0.0)

    def test_streak_boosts_mushroom_body_approach_synapses(self):
        """Validates that positive dopamine from clean exploration reinforces KC -> MBON-Approach synapses via STDP."""
        motion_metrics = {
            "dx": 0.0,
            "dy": 0.0,
            "flow": 0.005,
            "divergence": 0.002,
            "depth_balance": 0.1,
            "is_obstacle_close": False,
        }
        feat = self.mb.extract_feature_vector(motion_metrics, is_damage=False, red_excess=0.0)
        self.mb.step_sensory_projection(feat)

        w_appr_before = np.copy(self.mb.w_kc_appr)
        w_avoid_before = np.copy(self.mb.w_kc_avoid)

        # Apply streak reward (dopamine > 0.15)
        self.mb.apply_dopamine_plasticity(dopamine_level=1.2, escape_dir=1)

        # Active KCs must strengthen approach synapses and weaken avoid synapses
        active_kc = self.mb.last_active_kc_indices
        self.assertGreater(len(active_kc), 0)
        self.assertTrue(np.all(self.mb.w_kc_appr[active_kc] > w_appr_before[active_kc]))
        self.assertTrue(np.all(self.mb.w_kc_avoid[active_kc] < w_avoid_before[active_kc]))

    def test_da_decay_retention(self):
        """Validates that da_decay = 0.95 retains reward dopamine over corridor runs."""
        self.assertEqual(self.da.da_decay, 0.95)
        self.da.register_event("streak", magnitude=2.0)
        level_after_event = self.da.dopamine_level
        
        # Step once
        _, info = self.da.update(motion_flow=0.0002, is_obstacle=False)
        self.assertGreater(self.da.dopamine_level, 0.5)


if __name__ == "__main__":
    unittest.main()
