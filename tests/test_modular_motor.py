"""
test_modular_motor.py - Unit Test Suite for Modular DOOMFLY-Inspired Motor Pipeline
Validates NeuralDecoder, LocomotionController, and ReflexManager.
"""
import unittest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.platform.base import ActionKey
from core.platform.mock import MockInputDriver
from motor import NeuralDecoder, LocomotionController, ReflexManager, ReflexState


class TestModularMotor(unittest.TestCase):
    def test_decoder_ema_smoothing(self):
        """Verifies EMA low-pass filtering on noisy spike rates."""
        decoder = NeuralDecoder(tau_ms=100.0, turn_gain=1.5, forward_gain=1.5)

        # Step 1: Initial sudden spike
        res1 = decoder.decode(
            dnp20_left_rate=0.0,
            dnp20_right_rate=1.0,
            dnpe017_forward_rate=0.8,
            dnpe017_attack_rate=0.0,
            dt=0.02
        )
        # With tau=100ms and dt=20ms, alpha = 1 - exp(-0.2) ≈ 0.181
        self.assertGreater(res1["turn"], 0.1)
        self.assertLess(res1["turn"], 0.5)  # Should not jump to 1.0 instantly due to smoothing
        self.assertTrue(res1["forward"] > 0.1)
        self.assertFalse(res1["is_firing"])

        # Step 2: Sustained activation
        for _ in range(25):
            res_sustained = decoder.decode(
                dnp20_left_rate=0.0,
                dnp20_right_rate=1.0,
                dnpe017_forward_rate=0.8,
                dnpe017_attack_rate=0.5,
                dt=0.02
            )
        # Rates should have converged
        self.assertGreater(res_sustained["turn"], 0.8)
        self.assertGreater(res_sustained["forward"], 0.8)
        self.assertTrue(res_sustained["is_firing"])

    def test_locomotion_pure_steering(self):
        """Verifies corridor steering uses pure analog mouse yaw and does NOT press A/D strafe or arrow keys."""
        mock_driver = MockInputDriver(dry_run=True)
        controller = LocomotionController(mock_driver, mouse_gain_x=100.0, walk_threshold=0.1)

        # Apply left turn (-0.5) and forward walk (0.6)
        action = {"turn": -0.5, "forward": 0.6, "pitch": 0.0, "is_firing": False}
        info = controller.apply(action)

        self.assertTrue(controller.is_walking)
        self.assertIn(ActionKey.FORWARD, mock_driver.active_actions)

        # CRITICAL TEST (DOOMFLY Standard): Neither STRAFE nor arrow TURN keys should ever be pressed!
        self.assertNotIn(ActionKey.TURN_LEFT, mock_driver.active_actions)
        self.assertNotIn(ActionKey.TURN_RIGHT, mock_driver.active_actions)
        self.assertNotIn(ActionKey.STRAFE_LEFT, mock_driver.active_actions)
        self.assertNotIn(ActionKey.STRAFE_RIGHT, mock_driver.active_actions)

        # Check mouse move logged and slew-rate clamped to 16px to prevent camera whip
        moves = [act for act in mock_driver.logged_actions if act[0] == "move"]
        self.assertTrue(len(moves) > 0)
        self.assertEqual(moves[-1][1], -16)  # Slew-rate clamped to -16px for silky smooth camera

        # Now center steering
        action_straight = {"turn": 0.0, "forward": 0.6, "pitch": 0.0, "is_firing": False}
        controller.apply(action_straight)
        self.assertNotIn(ActionKey.TURN_LEFT, mock_driver.active_actions)
        self.assertTrue(controller.is_walking)

        # Stop walking
        action_stop = {"turn": 0.0, "forward": 0.0, "pitch": 0.0, "is_firing": False}
        controller.apply(action_stop)
        self.assertFalse(controller.is_walking)
        self.assertNotIn(ActionKey.FORWARD, mock_driver.active_actions)

    def test_locomotion_pitch_control(self):
        """Verifies pitch view angle produces vertical mouse dy (Issue #5)."""
        mock_driver = MockInputDriver(dry_run=True)
        controller = LocomotionController(mock_driver, mouse_gain_y=50.0)

        action = {"turn": 0.0, "forward": 0.0, "pitch": 0.4, "is_firing": False}
        controller.apply(action)

        moves = [act for act in mock_driver.logged_actions if act[0] == "move"]
        self.assertTrue(len(moves) > 0)
        self.assertEqual(moves[-1][2], 20)  # 0.4 * 50

    def test_reflex_manager_fsm(self):
        """Verifies emergency reflex overrides and clean recovery."""
        mock_driver = MockInputDriver(dry_run=True)
        reflexes = ReflexManager(mock_driver)

        self.assertFalse(reflexes.is_active())

        # Trigger obstacle saccade
        res = reflexes.trigger_obstacle_saccade(direction=1, flick_pixels=500)
        self.assertTrue(reflexes.is_active())
        self.assertEqual(reflexes.current_state, ReflexState.OBSTACLE_SACCADE)
        self.assertIn(ActionKey.BACKWARD, mock_driver.active_actions)
        self.assertIn(ActionKey.STRAFE_RIGHT, mock_driver.active_actions)

        # Advance timer past expiration
        now = time.time()
        is_still_active = reflexes.update(now=now + 0.5)
        self.assertFalse(is_still_active)
        self.assertFalse(reflexes.is_active())
        self.assertEqual(len(mock_driver.active_actions), 0)


if __name__ == "__main__":
    unittest.main()
