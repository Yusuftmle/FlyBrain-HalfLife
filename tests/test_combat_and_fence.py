"""
test_combat_and_fence.py - Unit tests for:
  1. Fence / Railing Low Obstacle Detection & Walkable Depth Balance
  2. Combat Retaliation 180° Whip Turn, Counter-Fire, and Jump-Strafe Evasion
"""
import unittest
import numpy as np
import time

from vision_bridge import VisionBridge
from motor.reflexes import ReflexManager, ReflexState
from core.platform.base import BaseInputDriver, ActionKey


class MockDriver(BaseInputDriver):
    def __init__(self):
        super().__init__()
        self.mouse_moves = []
        self.clicks = []
        self.pressed = set()

    def press_action(self, action: ActionKey):
        self.pressed.add(action)

    def release_action(self, action: ActionKey):
        self.pressed.discard(action)

    def mouse_move_relative(self, dx: int, dy: int):
        self.mouse_moves.append((dx, dy))

    def mouse_click(self, duration: float = 0.05):
        self.clicks.append(duration)

    def release_all(self):
        self.pressed.clear()

    def is_cursor_visible(self) -> bool:
        return False


class TestCombatAndFence(unittest.TestCase):

    def test_wall_obstacle_detection(self):
        """Validates that a flat wall ahead is detected as an obstacle."""
        bridge = VisionBridge(grid_width=60, grid_height=60)
        frame_wall = np.full((480, 640, 3), 110, dtype=np.uint8)
        currents, metrics = bridge.process_frame(frame_wall)
        self.assertTrue(metrics["is_obstacle_close"], "Flush flat wall ahead must trigger is_obstacle_close")

    def test_combat_retaliation_whip_and_fire(self):
        """Validates that taking damage triggers a 180° whip turn, sustained weapon counter-fire, and jump-strafe evasion."""
        driver = MockDriver()
        reflexes = ReflexManager(input_driver=driver)

        # Trigger retaliation turning to the right
        res = reflexes.trigger_combat_retaliation(direction=1, turn_whip=True)

        self.assertEqual(reflexes.current_state, ReflexState.COMBAT_RETALIATION)
        self.assertTrue(res["is_firing"])
        self.assertGreater(len(driver.mouse_moves), 0, "Must execute mouse whip turn")
        self.assertEqual(driver.mouse_moves[0][0], 200, "Must whip turn in the designated escape direction")
        self.assertGreater(len(driver.clicks), 0, "Must click primary fire to retaliate")
        self.assertGreaterEqual(driver.clicks[0], 0.08, "Weapon discharge must have sustained duration >= 80ms")
        self.assertIn(ActionKey.BACKWARD, driver.pressed, "Must back away from attacker")
        self.assertIn(ActionKey.JUMP, driver.pressed, "Must jump-dodge incoming enemy fire")
        self.assertIn(ActionKey.STRAFE_RIGHT, driver.pressed, "Must strafe away")

    def test_walkable_depth_balance_steers_to_opening(self):
        """Validates that depth_balance correctly favors the open walkable gap instead of a barrier."""
        bridge = VisionBridge(grid_width=60, grid_height=60)

        # Frame with dark barrier covering the LEFT side, but OPEN corridor on the RIGHT side
        frame = np.full((480, 640, 3), 140, dtype=np.uint8)
        frame[:, :320] = [30, 30, 30]

        currents, metrics = bridge.process_frame(frame)
        self.assertGreater(metrics["depth_balance"], 0.02, "Must steer right away from the left barrier into opening")

    def test_rpg_laser_not_mistaken_for_damage(self):
        """Validates that a bright red RPG laser dot on a wall does NOT trigger damage flash."""
        bridge = VisionBridge(grid_width=60, grid_height=60)
        frame_base = np.full((480, 640, 3), 120, dtype=np.uint8)
        bridge.detect_damage_flash(frame_base)

        # Draw a small intense red laser dot in the center of the screen
        import cv2
        frame_laser = frame_base.copy()
        cv2.circle(frame_laser, (320, 240), 8, (0, 0, 255), -1)

        is_dmg, mag = bridge.detect_damage_flash(frame_laser)
        self.assertFalse(is_dmg, "RPG laser dot on a wall must NOT be misclassified as damage")

    def test_screen_fade_damage_flash_detected(self):
        """Validates that a true GoldSrc full-screen red ScreenFade triggers damage detection."""
        bridge = VisionBridge(grid_width=60, grid_height=60)
        frame_base = np.full((480, 640, 3), 100, dtype=np.uint8)
        bridge.detect_damage_flash(frame_base)

        # Sudden full-screen red tint (damage flash)
        frame_fade = frame_base.copy()
        frame_fade[:, :, 2] = np.clip(frame_fade[:, :, 2].astype(int) + 140, 0, 255).astype(np.uint8)

        is_dmg, mag = bridge.detect_damage_flash(frame_fade)
        self.assertTrue(is_dmg, "Full-screen red ScreenFade must be recognized as damage")
        self.assertGreater(mag, 0.20)

    def test_adaptive_slew_rate_saccadic_turn(self):
        """Validates that obstacle evasion activates sharp saccadic turning (32 px/step) while normal is 16 px/step."""
        from core.platform.mock import MockInputDriver
        from motor.locomotion import LocomotionController

        driver = MockInputDriver(dry_run=True)
        controller = LocomotionController(driver, mouse_gain_x=100.0)

        # Normal turn without obstacle -> clamped to 16 px/frame
        controller.apply({"turn": 1.0, "is_obstacle_close": False})
        normal_move = [a for a in driver.logged_actions if a[0] == "move"][-1][1]
        self.assertEqual(normal_move, 16)

        # Reset controller state
        controller.last_mouse_dx = 0
        controller.mouse_accum_x = 0.0

        # Evasion turn with obstacle close -> saccadic boost to 32 px/frame
        controller.apply({"turn": 1.0, "is_obstacle_close": True})
        evade_move = [a for a in driver.logged_actions if a[0] == "move"][-1][1]
        self.assertEqual(evade_move, 32)


if __name__ == "__main__":
    unittest.main()
