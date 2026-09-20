"""
test_decision_sampler.py - Unit tests for:
  1. MochisRice-inspired 100ms Decision Sampling Window (10 Hz).
  2. Directional Hysteresis preventing sensorimotor limit-cycle camera shaking.
  3. Pure analog mouse yaw steering without keyboard arrow key chatter.
"""
import unittest
import numpy as np

from motor.decoder import NeuralDecoder
from motor.locomotion import LocomotionController
from core.platform.mock import MockInputDriver
from core.platform.base import ActionKey


class TestDecisionSampler(unittest.TestCase):

    def setUp(self):
        self.decoder = NeuralDecoder(
            tau_ms=100.0,
            turn_gain=3.5,
            turn_deadzone=0.035,
            decision_interval=0.10
        )
        self.mock_driver = MockInputDriver(dry_run=True)
        self.loco = LocomotionController(
            input_driver=self.mock_driver,
            mouse_gain_x=75.0,
            turn_deadzone=0.035
        )

    def test_decision_holds_for_100ms_window(self):
        """Verifies that an active turn decision is held and does not jitter within 100ms."""
        # Initial frame at t=0.00s with strong right turn (DNp20 Right > Left)
        res1 = self.decoder.decode(
            dnp20_left_rate=0.0,
            dnp20_right_rate=1.0,
            dnpe017_forward_rate=0.5,
            dnpe017_attack_rate=0.0,
            dt=0.02,
            now=0.00
        )
        self.assertGreater(res1["turn"], 0.0, "Must initiate right turn")
        self.assertEqual(self.decoder.active_turn_dir, 1)

        # Micro-fluctuation at t=0.03s (30ms later, within 100ms window) with sudden left spike
        res2 = self.decoder.decode(
            dnp20_left_rate=0.8,
            dnp20_right_rate=0.2,
            dnpe017_forward_rate=0.5,
            dnpe017_attack_rate=0.0,
            dt=0.02,
            now=0.03
        )
        # Decision must remain RIGHT because 100ms window has not elapsed!
        self.assertEqual(self.decoder.active_turn_dir, 1, "Decision must be locked within 100ms window")
        self.assertGreater(res2["turn"], 0.0, "Turn action must not violently flip sign")

    def test_directional_hysteresis_damps_noise(self):
        """Verifies that noise hovering around deadzone cannot flip active turn direction."""
        # Sinek is steering right
        self.decoder.decode(0.0, 1.0, 0.5, 0.0, dt=0.02, now=0.00)
        self.assertEqual(self.decoder.active_turn_dir, 1)

        # At t=0.12s (after decision interval), sensory input hovers near neutral with slight left noise
        # raw_turn_diff = -0.02 (which is smaller than deadzone * 1.4 = 0.049)
        self.decoder.smooth_dnp20_r = 0.08
        self.decoder.smooth_dnp20_l = 0.10  # diff = -0.02
        res = self.decoder.decode(0.10, 0.08, 0.5, 0.0, dt=0.02, now=0.12)

        # Hysteresis must NOT flip to left on weak opposing noise!
        self.assertNotEqual(self.decoder.active_turn_dir, -1, "Small noise must not flip active direction to left")

    def test_pure_analog_mouse_steering_no_arrow_keys(self):
        """Verifies that steering never presses keyboard arrow keys (+left / +right)."""
        action = {"turn": 0.8, "forward": 0.5, "pitch": 0.0, "is_firing": False}
        self.loco.apply(action)

        self.assertNotIn(ActionKey.TURN_LEFT, self.mock_driver.active_actions)
        self.assertNotIn(ActionKey.TURN_RIGHT, self.mock_driver.active_actions)
        self.assertNotIn(ActionKey.STRAFE_LEFT, self.mock_driver.active_actions)
        self.assertNotIn(ActionKey.STRAFE_RIGHT, self.mock_driver.active_actions)

        # Ensure mouse delta is clamped to max 16-22 px/frame
        moves = [act for act in self.mock_driver.logged_actions if act[0] == "move"]
        self.assertTrue(len(moves) > 0)
        self.assertLessEqual(abs(moves[-1][1]), 22)


if __name__ == "__main__":
    unittest.main()
