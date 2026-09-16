"""
base.py - Hardware Abstraction Layer (HAL) Interfaces and Semantic Enums
Defines platform-agnostic contracts for screen capture and hardware input injection.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np


class ActionKey(Enum):
    FORWARD = "forward"          # W / Move Forward
    BACKWARD = "backward"        # S / Move Backward
    STRAFE_LEFT = "strafe_left"  # A / Move Left
    STRAFE_RIGHT = "strafe_right"# D / Move Right
    TURN_LEFT = "turn_left"      # Left Arrow / Turn Left
    TURN_RIGHT = "turn_right"    # Right Arrow / Turn Right
    JUMP = "jump"                # Space / Jump
    CROUCH = "crouch"            # Left Ctrl / Crouch
    FIRE = "fire"                # Left Mouse Button / Primary Attack
    QUICKLOAD = "quickload"      # F9 / Quick Load


class BaseScreenCapture(ABC):
    """Abstract Base Class for OS-level game window frame capture."""

    @abstractmethod
    def find_target_window(self, titles: Optional[List[str]] = None, process_names: Optional[List[str]] = None) -> bool:
        """Locates the target game window handle/ID."""
        pass

    @abstractmethod
    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Grabs active game frame with ultra-low latency.
        Returns BGR numpy array (H, W, 3) or None on capture failure.
        """
        pass

    @abstractmethod
    def is_game_running(self) -> bool:
        """Returns True if the target game window is active and valid."""
        pass

    @abstractmethod
    def is_game_process_running(self) -> bool:
        """Returns True if the target game process is running on the OS."""
        pass

    @abstractmethod
    def focus_game_window(self) -> bool:
        """Brings the game window to the foreground."""
        pass

    @abstractmethod
    def is_game_focused(self) -> bool:
        """Returns True if the target game window is active in the foreground."""
        pass

    @abstractmethod
    def close(self):
        """Releases all graphics and OS resources."""
        pass


class BaseInputDriver(ABC):
    """Abstract Base Class for OS-level hardware input injection."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.active_actions = set()

    @abstractmethod
    def press_action(self, action: ActionKey):
        """Holds down key/button corresponding to the semantic action."""
        pass

    @abstractmethod
    def release_action(self, action: ActionKey):
        """Releases key/button corresponding to the semantic action."""
        pass

    def tap_action(self, action: ActionKey, duration: float = 0.05):
        """Taps an action for a short duration."""
        import time
        self.press_action(action)
        time.sleep(duration)
        self.release_action(action)

    @abstractmethod
    def mouse_click(self, duration: float = 0.05):
        """Sends primary mouse button click (Fire weapon)."""
        pass

    @abstractmethod
    def mouse_move_relative(self, dx: int, dy: int):
        """Moves mouse pointer by relative delta (Yaw/Pitch rotation)."""
        pass

    @abstractmethod
    def release_all(self):
        """Emergency failsafe: releases all currently held keys."""
        pass

    @abstractmethod
    def is_cursor_visible(self) -> bool:
        """Checks if OS mouse cursor is visible (indicates pause menu / inventory)."""
        pass
