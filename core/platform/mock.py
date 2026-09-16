"""
mock.py - Headless and Simulation Mock Drivers
Allows running full FlyBrain simulation, testing, and CI without a physical display or game process.
"""
import logging
from typing import Optional, List
import numpy as np

from .base import BaseScreenCapture, BaseInputDriver, ActionKey

logger = logging.getLogger("FlyBrain.Platform.Mock")


class MockScreenCapture(BaseScreenCapture):
    """Generates synthetic frames for unit tests, headless Docker, and CI."""
    def __init__(self, target_title: str = "MockGame"):
        self.target_title = target_title
        self.frame_count = 0

    def find_target_window(self, titles: Optional[List[str]] = None, process_names: Optional[List[str]] = None) -> bool:
        return True

    def grab_frame(self) -> Optional[np.ndarray]:
        """Returns synthetic 640x480 test pattern with motion."""
        self.frame_count += 1
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add dynamic synthetic gradient to simulate visual stimulation
        shift = (self.frame_count * 4) % 640
        frame[:, shift:shift+50, :] = 200
        return frame

    def is_game_running(self) -> bool:
        return True

    def is_game_process_running(self) -> bool:
        return True

    def focus_game_window(self) -> bool:
        return True

    def is_game_focused(self) -> bool:
        return True

    def close(self):
        pass


class MockInputDriver(BaseInputDriver):
    """Simulated input driver that logs actions without sending OS events."""
    def __init__(self, dry_run: bool = True):
        super().__init__(dry_run=True)
        self.logged_actions = []

    def press_action(self, action: ActionKey):
        self.active_actions.add(action)
        self.logged_actions.append(("press", action.value))

    def release_action(self, action: ActionKey):
        self.active_actions.discard(action)
        self.logged_actions.append(("release", action.value))

    def mouse_click(self, duration: float = 0.05):
        self.logged_actions.append(("click", duration))

    def mouse_move_relative(self, dx: int, dy: int):
        self.logged_actions.append(("move", dx, dy))

    def release_all(self):
        self.active_actions.clear()

    def is_cursor_visible(self) -> bool:
        return False
