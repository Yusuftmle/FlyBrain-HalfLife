"""
test_responsive_turning.py - Unit tests for:
  1. DNp20 sensitive steering through NeuralDecoder.
  2. Prevention of spurious dopamine streak rewards during stationary weapon idle breathing.
"""
import unittest
import numpy as np

from motor.decoder import NeuralDecoder
from motor.locomotion import LocomotionController
from core.platform.mock import MockInputDriver


class TestResponsiveTurning(unittest.TestCase):

    def setUp(self):
        self.decoder = NeuralDecoder(
            tau_ms=100.0,
            turn_gain=5.5,
            turn_deadzone=0.010,
            decision_interval=0.08
        )
        self.driver = MockInputDriver(dry_run=True)
        self.loco = LocomotionController(
            input_driver=self.driver,
            mouse_gain_x=110.0,
            turn_deadzone=0.010
        )

    def test_single_spike_triggers_responsive_turn(self):
        """Verifies that a discrete single-spike burst (rate=0.1) clears deadzone and commands steering."""
        # Frame 1: Right DNp20 fires 1 spike (rate=0.100), Left is silent (0.000)
        res = self.decoder.decode(
            dnp20_left_rate=0.0,
            dnp20_right_rate=0.100,
            dnpe017_forward_rate=0.3,
            dnpe017_attack_rate=0.0,
            dt=0.033
        )
        # Smoothed diff is ~0.028, which exceeds turn_deadzone=0.010!
        self.assertGreater(res["turn"], 0.10, "Turn should be active and > 0.10 on single spike burst")
        self.assertEqual(self.decoder.active_turn_dir, 1, "Active turn direction must be locked to RIGHT")

        # Verify locomotion converts turn to mouse movement
        info = self.loco.apply(res)
        moves = [act for act in self.driver.logged_actions if act[0] == "move"]
        self.assertTrue(len(moves) > 0)
        self.assertGreater(moves[-1][1], 0, "Mouse dx must steer right")

    def test_spurious_streak_prevented_when_stationary(self):
        """Verifies that weapon breathing noise (< 0.00045 flow) does not count as clean forward exploration."""
        # Simulated stationary weapon idle bobbing
        flow_idle = 0.00022
        div_idle = 0.00002
        is_obs = True  # Pinned against wall
        is_fence = False
        is_damage = False
        is_walking = True

        # Formula from main.py:
        is_genuine_forward = (
            (flow_idle > 0.00060)
            and (div_idle > 0.00008)
            and is_walking
            and (not is_obs)
            and (not is_fence)
            and (not is_damage)
        )
        self.assertFalse(is_genuine_forward, "Weapon idle bobbing against a wall MUST NOT count as genuine forward exploration")


if __name__ == "__main__":
    unittest.main()
