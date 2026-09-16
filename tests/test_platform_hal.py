"""
test_platform_hal.py - Unit Test Suite for Hardware Abstraction Layer (HAL)
Validates cross-platform drivers, lazy factory dispatch, and bridge delegation.
"""
import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.platform.base import ActionKey, BaseScreenCapture, BaseInputDriver
from core.platform import get_screen_capture_driver, get_input_driver, get_platform_drivers
from core.platform.mock import MockScreenCapture, MockInputDriver
from vision_bridge import VisionBridge
from input_bridge import InputBridge


class TestPlatformHAL(unittest.TestCase):
    def test_action_key_definitions(self):
        """Verifies semantic action keys."""
        required_actions = [
            ActionKey.FORWARD, ActionKey.BACKWARD, ActionKey.STRAFE_LEFT,
            ActionKey.STRAFE_RIGHT, ActionKey.TURN_LEFT, ActionKey.TURN_RIGHT,
            ActionKey.JUMP, ActionKey.CROUCH, ActionKey.FIRE, ActionKey.QUICKLOAD
        ]
        for act in required_actions:
            self.assertIsInstance(act.value, str)

    def test_mock_drivers(self):
        """Tests MockScreenCapture and MockInputDriver."""
        cap = MockScreenCapture()
        inp = MockInputDriver(dry_run=True)

        self.assertTrue(isinstance(cap, BaseScreenCapture))
        self.assertTrue(isinstance(inp, BaseInputDriver))

        # Test frame grab
        frame = cap.grab_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

        # Test input actions
        inp.press_action(ActionKey.FORWARD)
        inp.release_action(ActionKey.FORWARD)
        inp.mouse_move_relative(10, -5)
        inp.mouse_click()
        inp.release_all()

        self.assertEqual(len(inp.active_actions), 0)
        self.assertIn(("press", ActionKey.FORWARD.value), inp.logged_actions)
        self.assertIn(("release", ActionKey.FORWARD.value), inp.logged_actions)
        self.assertIn(("move", 10, -5), inp.logged_actions)

    def test_factory_dispatch_override(self):
        """Tests dynamic factory with FLYBRAIN_PLATFORM=mock override."""
        old_val = os.environ.get("FLYBRAIN_PLATFORM")
        try:
            os.environ["FLYBRAIN_PLATFORM"] = "mock"
            cap, inp = get_platform_drivers(dry_run=True)
            self.assertIsInstance(cap, MockScreenCapture)
            self.assertIsInstance(inp, MockInputDriver)
        finally:
            if old_val is not None:
                os.environ["FLYBRAIN_PLATFORM"] = old_val
            else:
                os.environ.pop("FLYBRAIN_PLATFORM", None)

    def test_current_platform_drivers(self):
        """Tests drivers loaded on current host."""
        cap, inp = get_platform_drivers(dry_run=True)
        self.assertTrue(isinstance(cap, BaseScreenCapture))
        self.assertTrue(isinstance(inp, BaseInputDriver))

        # Test safe non-crashing methods
        self.assertIsInstance(cap.is_game_running(), bool)
        self.assertIsInstance(cap.is_game_process_running(), bool)
        self.assertIsInstance(inp.is_cursor_visible(), bool)
        inp.release_all()

    def test_linux_module_integrity(self):
        """Verifies Linux driver classes can be imported and inspected without errors."""
        from core.platform.linux import LinuxScreenCapture, LinuxInputDriver
        self.assertTrue(issubclass(LinuxScreenCapture, BaseScreenCapture))
        self.assertTrue(issubclass(LinuxInputDriver, BaseInputDriver))

        # Initialize LinuxInputDriver in dry_run mode
        lin_inp = LinuxInputDriver(dry_run=True)
        lin_inp.press_action(ActionKey.FORWARD)
        lin_inp.release_action(ActionKey.FORWARD)
        lin_inp.mouse_move_relative(5, 5)
        lin_inp.mouse_click()
        lin_inp.release_all()

    def test_bridge_integration(self):
        """Verifies that VisionBridge and InputBridge properly integrate with HAL."""
        vision = VisionBridge(grid_width=30, grid_height=30)
        input_b = InputBridge(dry_run=True)

        self.assertTrue(hasattr(vision, "capture_driver"))
        self.assertTrue(hasattr(input_b, "input_driver"))

        # Test properties and methods
        self.assertTrue(isinstance(vision.capture_driver, BaseScreenCapture))
        self.assertTrue(isinstance(input_b.input_driver, BaseInputDriver))

        # Vision grab should return valid numpy array
        frame = vision.capture_frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(len(frame.shape), 3)

        # Input dispatch test
        res = input_b.decode_and_dispatch(
            dnp20_left_rate=0.1,
            dnp20_right_rate=0.0,
            dnpe017_forward_rate=0.2,
            dnpe017_attack_rate=0.0
        )
        self.assertIn("action", res)
        input_b.release_all()


if __name__ == "__main__":
    unittest.main()
