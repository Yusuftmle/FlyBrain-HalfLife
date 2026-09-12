"""
Windows DirectInput Scancode Simulator
Uses ctypes SendInput for Full DirectX / GoldSrc (Half-Life) Compatibility
"""
import ctypes
import time
from typing import Optional

# DirectInput Hardware Scancodes (DirectX / GoldSrc standards)
DIK_W = 0x11
DIK_A = 0x1E
DIK_S = 0x1F
DIK_D = 0x20
DIK_SPACE = 0x39
DIK_LCONTROL = 0x1D

# Windows SendInput C Structure Definitions
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I)
    ]

KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

class DirectInputDriver:
    """
    Sends hardware-level DirectInput scancodes to games.
    Supports dry-run mode to verify simulation safely without taking over OS input.
    """
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.active_keys = set()
        self.last_action_str = "IDLE"

    def press_key(self, hex_key_code: int):
        """Holds down hardware key scancode."""
        self.active_keys.add(hex_key_code)
        if self.dry_run:
            return
        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, hex_key_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception:
            pass

    def release_key(self, hex_key_code: int):
        """Releases hardware key scancode."""
        self.active_keys.discard(hex_key_code)
        if self.dry_run:
            return
        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, hex_key_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception:
            pass

    def tap_key(self, hex_key_code: int, duration: float = 0.05):
        """Taps a key for a short duration."""
        self.press_key(hex_key_code)
        time.sleep(duration)
        self.release_key(hex_key_code)

    def mouse_click(self, duration: float = 0.05):
        """Sends primary mouse click (Fire weapon)."""
        self.last_action_str = "FIRE!"
        if self.dry_run:
            return
        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            time.sleep(duration)
            ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception:
            pass

    def mouse_move_relative(self, dx: int, dy: int):
        """Moves mouse pointer by relative delta (Yaw/Pitch view rotation)."""
        if self.dry_run:
            return
        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception:
            pass

    def release_all(self):
        """Releases any currently held keys."""
        for key in list(self.active_keys):
            self.release_key(key)
        self.active_keys.clear()
