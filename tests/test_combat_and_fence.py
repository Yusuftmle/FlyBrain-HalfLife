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

    def test_fence_obstacle_detection(self):
        """Validates that half-fences and low barriers in the lower frontal field are detected as obstacles."""
        bridge = VisionBridge(grid_width=60, grid_height=60)

        # Create a frame with clear open ceiling (top 24 rows) but a dense fence in the lower field (rows 25..55)
        frame = np.full((480, 640, 3), 180, dtype=np.uint8) # Sky/Ceiling
        # Draw repetitive fence grid in the bottom half
        for y in range(240, 440, 15):
            frame[y : y + 2, 160:480] = [30, 30, 30]
        for x in range(160, 480, 12):
            frame[240:440, x : x + 2] = [30, 30, 30]

        currents, metrics = bridge.process_frame(frame)
        self.assertTrue(metrics["is_fence"], "Dense repetitive lower barrier must be flagged as is_fence")
        self.assertTrue(metrics["is_obstacle_close"], "Fence ahead must trigger is_obstacle_close")

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
        """Validates that depth_balance correctly favors the open walkable gap instead of a fence."""
        bridge = VisionBridge(grid_width=60, grid_height=60)

        # Frame with a fence covering the LEFT side, but OPEN corridor on the RIGHT side
        frame = np.full((480, 640, 3), 140, dtype=np.uint8)
        # Left half fence
        for y in range(200, 440, 12):
            frame[y : y + 3, :320] = [10, 10, 10]
        for x in range(0, 320, 10):
            frame[200:440, x : x + 3] = [10, 10, 10]

        currents, metrics = bridge.process_frame(frame)
        # Left is cluttered with fence -> depth_balance must be POSITIVE (steer right into open gap)
        self.assertGreater(metrics["depth_balance"], 0.02, "Must steer right away from the left fence into opening")


if __name__ == "__main__":
    unittest.main()
