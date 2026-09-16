"""
direct_input.py - Platform-Agnostic Input Simulator (Compatibility Layer)
Delegates to core.platform HAL drivers (Windows DirectInput / Linux evdev/uinput).
"""
import time
from typing import Optional
from core.platform import get_input_driver, ActionKey

# DirectInput Hardware Scancodes (DirectX / GoldSrc standards)
DIK_W = 0x11
DIK_A = 0x1E
DIK_S = 0x1F
DIK_D = 0x20
DIK_SPACE = 0x39
DIK_LCONTROL = 0x1D

SCANCODE_TO_ACTION = {
    DIK_W: ActionKey.FORWARD,
    DIK_S: ActionKey.BACKWARD,
    DIK_A: ActionKey.STRAFE_LEFT,
    DIK_D: ActionKey.STRAFE_RIGHT,
    DIK_SPACE: ActionKey.JUMP,
    DIK_LCONTROL: ActionKey.CROUCH
}


class DirectInputDriver:
    """
    Sends hardware-level input scancodes to games via core.platform HAL.
    Supports dry-run mode to verify simulation safely without taking over OS input.
    """
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.active_keys = set()
        self.last_action_str = "IDLE"
        self.driver = get_input_driver(dry_run=dry_run)

    def press_key(self, hex_key_code: int):
        """Holds down hardware key scancode."""
        self.active_keys.add(hex_key_code)
        action = SCANCODE_TO_ACTION.get(hex_key_code)
        if action:
            self.driver.press_action(action)

    def release_key(self, hex_key_code: int):
        """Releases hardware key scancode."""
        self.active_keys.discard(hex_key_code)
        action = SCANCODE_TO_ACTION.get(hex_key_code)
        if action:
            self.driver.release_action(action)

    def tap_key(self, hex_key_code: int, duration: float = 0.05):
        """Taps a key for a short duration."""
        self.press_key(hex_key_code)
        time.sleep(duration)
        self.release_key(hex_key_code)

    def mouse_click(self, duration: float = 0.05):
        """Sends primary mouse click (Fire weapon)."""
        self.last_action_str = "FIRE!"
        self.driver.mouse_click(duration)

    def mouse_move_relative(self, dx: int, dy: int):
        """Moves mouse pointer by relative delta (Yaw/Pitch view rotation)."""
        self.driver.mouse_move_relative(dx, dy)

    def release_all(self):
        """Releases any currently held keys."""
        self.driver.release_all()
        self.active_keys.clear()
