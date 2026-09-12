"""
input_bridge.py - Descending Motor Neuron Decoder & DirectX DirectInput Driver
Translates biological DNp20 and DNpe017 firing patterns into hardware scancodes
with refractory period cooldown to prevent key spamming and buffer overflow.
"""
import logging
import time
import ctypes
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger("FlyBrain.InputBridge")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# High-Performance DirectInput API (DirectX / GoldSrc compatibility)
try:
    import pydirectinput
    pydirectinput.PAUSE = 0.0 # Crucial: prevent default 100ms lag on every keypress
    pydirectinput.FAILSAFE = False
    PYDIRECTINPUT_AVAILABLE = True
except ImportError:
    PYDIRECTINPUT_AVAILABLE = False

# Win32 DirectInput Hardware Scancodes (Hardware-level keyboard/mouse fallback)
DIK_W = 0x11
DIK_A = 0x1E
DIK_S = 0x1F
DIK_D = 0x20
DIK_SPACE = 0x39
DIK_LCONTROL = 0x1D
DIK_LEFT = 0xCB   # DirectInput Arrow Left
DIK_RIGHT = 0xCD  # DirectInput Arrow Right

PUL = ctypes.POINTER(ctypes.c_ulong)
import ctypes.wintypes

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("hCursor", ctypes.c_void_p),
        ("ptScreenPos", ctypes.wintypes.POINT)
    ]

VK_F9 = 0x78
VK_F10 = 0x79
VK_F12 = 0x7B

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

class InputBridge:
    """
    Decodes Drosophila Premotor Descending Neurons into FPS DirectInput commands.
    Enforces a strict refractory cooldown (e.g. 100ms) to prevent game buffer locking.
    Includes active window focus lock, in-game menu/console detection, and F10 emergency toggle.
    """
    def __init__(
        self, 
        turn_threshold: float = 0.03,
        walk_threshold: float = 0.05,
        attack_threshold: float = 0.35,
        press_duration: float = 0.06,
        cooldown_duration: float = 0.08,
        mouse_turn_gain: int = 16,
        dry_run: bool = False,
        target_hwnd: Optional[int] = None
    ):
        self.turn_threshold: float = turn_threshold
        self.walk_threshold: float = walk_threshold
        self.attack_threshold: float = attack_threshold
        self.press_duration: float = press_duration
        self.cooldown_duration: float = cooldown_duration
        self.mouse_turn_gain: int = mouse_turn_gain
        self.dry_run: bool = dry_run
        self.target_hwnd: Optional[int] = target_hwnd
        
        # User pause & safety locks
        self.user_paused: bool = False
        self.menu_lock_enabled: bool = False
        self.pause_reason: str = ""
        self._last_hotkey_time: float = 0.0
        
        self.active_keys: set = set()
        self.last_press_time: Dict[str, float] = {
            "turn_l": 0.0,
            "turn_r": 0.0,
            "forward": 0.0,
            "back": 0.0,
            "attack": 0.0
        }
        self.is_walking: bool = False
        self.is_stepping_back: bool = False
        self.is_turning_l: bool = False
        self.is_turning_r: bool = False
        
        self.current_action: str = "IDLE"
        self.is_firing: bool = False
        self.is_escaping: bool = False
        self.last_escape_time: float = 0.0
        self.last_fire_time: float = 0.0
        self.last_combat_retaliation_time: float = 0.0
        logger.info(f"InputBridge initialized (Dry-Run: {self.dry_run}, Cooldown: {self.cooldown_duration*1000:.0f}ms, MouseGain: {self.mouse_turn_gain}).")
        logger.info("🔒 Safety Guard engaged: F10/F12 to Toggle Bot, F9 to Toggle Menu Lock, Desktop auto-lock active.")

    def is_allowed_overlay_or_telemetry(self, fg: int) -> bool:
        """Checks if the foreground window is NVIDIA GeForce Overlay or FlyBrain Telemetry window."""
        if not fg:
            return False
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
            hProc = kernel32.OpenProcess(0x1000, False, pid.value)
            pname = ""
            if hProc:
                buf = ctypes.create_unicode_buffer(1024)
                sz = ctypes.wintypes.DWORD(1024)
                if kernel32.QueryFullProcessImageNameW(hProc, 0, buf, ctypes.byref(sz)):
                    pname = buf.value.lower()
                kernel32.CloseHandle(hProc)

            length = user32.GetWindowTextLengthW(fg)
            title = ""
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(fg, buff, length + 1)
                title = buff.value.lower()

            # NVIDIA GeForce Experience / ShadowPlay / Share Overlay
            if any(k in pname for k in ['nvcontainer', 'nvidia share', 'shadowplay', 'nvidia geforce overlay']) or \
               any(k in title for k in ['nvidia', 'geforce overlay', 'shadowplay']):
                return True

            # FlyBrain Telemetry Window
            if any(k in pname for k in ['python.exe', 'pythonw.exe']) and \
               any(k in title for k in ['flybrain', 'telemetry', 'connectome']):
                return True

            return False
        except Exception:
            return False

    def is_game_focused(self) -> bool:
        """Verifies if the active foreground window is Half-Life / target game window or allowed overlay."""
        if self.dry_run:
            return True
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            try:
                hinput = user32.OpenInputDesktop(0, False, 0x01FF)
                if hinput:
                    user32.SetThreadDesktop(hinput)
            except Exception:
                pass

            fg = user32.GetForegroundWindow()
            if not fg:
                return False
            if self.target_hwnd and self.target_hwnd != 0 and fg == self.target_hwnd:
                return True

            # Allow NVIDIA GeForce Overlay (Alt+Z) & FlyBrain Telemetry Window without declaring lockout
            if self.is_allowed_overlay_or_telemetry(fg):
                return True

            # Check process image of foreground window
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
            hProc = kernel32.OpenProcess(0x1000, False, pid.value)
            if hProc:
                buf = ctypes.create_unicode_buffer(1024)
                sz = ctypes.wintypes.DWORD(1024)
                if kernel32.QueryFullProcessImageNameW(hProc, 0, buf, ctypes.byref(sz)):
                    pname = buf.value.lower()
                    if any(k in pname for k in ['hl.exe', 'cstrike.exe', 'half-life.exe', 'cs.exe']):
                        kernel32.CloseHandle(hProc)
                        self.target_hwnd = fg
                        return True
                kernel32.CloseHandle(hProc)

            # Fallback search by title / class
            length = user32.GetWindowTextLengthW(fg)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(fg, buff, length + 1)
                title = buff.value.lower()
                if any(k in title for k in ["half-life", "valve001", "hl.exe", "doom", "gzdoom"]):
                    self.target_hwnd = fg
                    return True
            return False
        except Exception:
            return False

    def is_cursor_visible(self) -> bool:
        """
        Detects if the Windows mouse cursor is showing while Half-Life is focused.
        In FPS gameplay, mouse cursor is hidden/locked.
        When ESC menu or ~ console is opened, cursor becomes visible.
        """
        if self.dry_run or not self.menu_lock_enabled:
            return False
        try:
            fg = ctypes.windll.user32.GetForegroundWindow()
            # If NVIDIA overlay or Telemetry window is open, cursor is expected; do not treat as game pause
            if self.is_allowed_overlay_or_telemetry(fg):
                return False

            if not self.is_game_focused():
                return False
            ci = CURSORINFO()
            ci.cbSize = ctypes.sizeof(CURSORINFO)
            if ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):
                # CURSOR_SHOWING = 0x00000001
                return bool(ci.flags & 1)
        except Exception:
            pass
        return False

    def check_hotkeys(self) -> bool:
        """Listens for F9 (Menu Lock toggle) and F10 / F12 (Bot pause toggle)."""
        try:
            user32 = ctypes.windll.user32
            now = time.time()
            if (now - self._last_hotkey_time) < 0.35:
                return False

            # F9: Toggle Menu Lock
            state_f9 = user32.GetAsyncKeyState(VK_F9)
            if (state_f9 & 0x8000) or (state_f9 & 0x0001):
                self._last_hotkey_time = now
                self.toggle_menu_lock()
                return True

            # F10 / F12: Toggle Pause
            for vk in [VK_F10, VK_F12]:
                state = user32.GetAsyncKeyState(vk)
                if (state & 0x8000) or (state & 0x0001):
                    self._last_hotkey_time = now
                    self.toggle_pause()
                    return True
        except Exception:
            pass
        return False

    def toggle_menu_lock(self, force_state: Optional[bool] = None) -> bool:
        """Toggles ESC/~ menu cursor lock."""
        if force_state is not None:
            self.menu_lock_enabled = force_state
        else:
            self.menu_lock_enabled = not self.menu_lock_enabled
        state_str = "AKTİF" if self.menu_lock_enabled else "DEVRE DIŞI"
        logger.info(f"🖱️ [MENU LOCK] Menü/Konsol Koruması: {state_str} (F9).")
        return self.menu_lock_enabled

    def toggle_pause(self, force_state: Optional[bool] = None) -> bool:
        """Toggles user-controlled pause state."""
        if force_state is not None:
            self.user_paused = force_state
        else:
            self.user_paused = not self.user_paused
        
        if self.user_paused:
            self.release_all()
            self.pause_reason = "USER_PAUSED"
            logger.info("⏸️ [PAUSE] Sinek DURDURULDU (F10 / Web Stop). Girisler donduruldu.")
        else:
            self.pause_reason = ""
            logger.info("▶️ [RESUME] Sinek AKTIF. Girisler devrede.")
        return self.user_paused

    def can_send_input(self) -> bool:
        """Master safety lock. Returns True only if safe to dispatch hardware inputs."""
        if self.user_paused:
            return False
        if self.dry_run:
            return True
        fg = ctypes.windll.user32.GetForegroundWindow()
        if self.is_allowed_overlay_or_telemetry(fg):
            return False
        if not self.is_game_focused():
            return False
        if self.is_cursor_visible():
            return False
        return True

    def _send_key_event(self, scancode: int, is_up: bool = False):
        if self.dry_run:
            return
        if not is_up and not self.can_send_input():
            return
        try:
            flags = KEYEVENTF_SCANCODE
            if is_up:
                flags |= KEYEVENTF_KEYUP
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, scancode, flags, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception as e:
            logger.error(f"Error sending key event (scancode {scancode}): {e}")

    def _send_mouse_click(self, duration: float = 0.001):
        if self.dry_run or not self.can_send_input():
            return
        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception as e:
            logger.error(f"Error sending mouse click: {e}")

    def _send_mouse_move(self, dx: int, dy: int):
        if self.dry_run or not self.can_send_input():
            return
        if PYDIRECTINPUT_AVAILABLE:
            try:
                pydirectinput.moveRel(int(dx), int(dy), relative=True)
                return
            except Exception as e:
                logger.debug(f"pydirectinput moveRel fallback: {e}")
        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception as e:
            logger.error(f"Error sending mouse move: {e}")

    def _send_key_down(self, key_name: str, scancode: int):
        if self.dry_run or not self.can_send_input():
            return
        if PYDIRECTINPUT_AVAILABLE:
            try:
                pydirectinput.keyDown(key_name)
                return
            except Exception as e:
                logger.debug(f"pydirectinput keyDown({key_name}) fallback: {e}")
        self._send_key_event(scancode, is_up=False)

    def _send_key_up(self, key_name: str, scancode: int):
        if self.dry_run:
            return
        if PYDIRECTINPUT_AVAILABLE:
            try:
                pydirectinput.keyUp(key_name)
                return
            except Exception as e:
                logger.debug(f"pydirectinput keyUp({key_name}) fallback: {e}")
        self._send_key_event(scancode, is_up=True)

    def _send_click(self, duration: Optional[float] = None):
        if self.dry_run or not self.can_send_input():
            return
        if PYDIRECTINPUT_AVAILABLE:
            try:
                pydirectinput.click(button='left')
                return
            except Exception as e:
                logger.debug(f"pydirectinput click fallback: {e}")
        self._send_mouse_click()

    def decode_and_dispatch(
        self, 
        dnp20_left_rate: float, 
        dnp20_right_rate: float, 
        dnpe017_forward_rate: float, 
        dnpe017_attack_rate: float,
        is_obstacle_close: bool = False,
        mdn_backward_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluates motor neuron firing rates, enforces refractory periods, and dispatches inputs
        using pydirectinput (or Win32 DirectInput scancodes) to control GoldSrc Half-Life.
        Includes biological Moonwalker Descending Neuron (MDN) backward locomotion (Bidaye et al. 2014).
        """
        now = time.time()
        
        # 0. Global Safety Guards: Check Hotkeys (F10/F12) & Window Focus & Menu Cursor
        self.check_hotkeys()
        
        if self.user_paused:
            self.release_all()
            self.current_action = "PAUSED [F10 / STOP BUTONU]"
            return {
                "turn_diff": 0.0,
                "forward_rate": 0.0,
                "backward_rate": 0.0,
                "attack_rate": 0.0,
                "action": self.current_action,
                "is_firing": False,
                "is_escaping": False,
                "is_paused": True,
                "pause_reason": "USER_PAUSED"
            }
            
        fg = ctypes.windll.user32.GetForegroundWindow()
        if self.is_allowed_overlay_or_telemetry(fg):
            self.release_all()
            self.current_action = "OBSERVING (OVERLAY / TELEMETRY ODAKTA)"
            return {
                "turn_diff": 0.0,
                "forward_rate": 0.0,
                "backward_rate": 0.0,
                "attack_rate": 0.0,
                "action": self.current_action,
                "is_firing": False,
                "is_escaping": False,
                "is_paused": False,
                "pause_reason": ""
            }

        if not self.is_game_focused():
            self.release_all()
            self.current_action = "PAUSED [ODAK YOK - MASAÜSTÜ]"
            return {
                "turn_diff": 0.0,
                "forward_rate": 0.0,
                "backward_rate": 0.0,
                "attack_rate": 0.0,
                "action": self.current_action,
                "is_firing": False,
                "is_escaping": False,
                "is_paused": True,
                "pause_reason": "WINDOW_UNFOCUSED"
            }

        if self.is_cursor_visible():
            self.release_all()
            self.current_action = "PAUSED [MENÜ / KONSOL AÇIK (~, ESC)]"
            return {
                "turn_diff": 0.0,
                "forward_rate": 0.0,
                "backward_rate": 0.0,
                "attack_rate": 0.0,
                "action": self.current_action,
                "is_firing": False,
                "is_escaping": False,
                "is_paused": True,
                "pause_reason": "MENU_ACTIVE"
            }
        
        turn_diff = float(dnp20_left_rate - dnp20_right_rate)
        actions = []

        # 1. DNp20 Steering Decoder (Half-Life GoldSrc: A / D strafe + Active Mouse Look + Arrow Keys)
        # Mouse movement and arrow keys provide decisive, biological orientation towards open corridors
        gain = int(max(45, min(140, abs(turn_diff) * 160.0)))
        if turn_diff > self.turn_threshold:
            mouse_dx = -gain
            self._send_mouse_move(dx=mouse_dx, dy=0)
            if not self.is_turning_l or (now - self.last_press_time["turn_l"]) > self.cooldown_duration:
                self._send_key_up('d', DIK_D)
                self._send_key_up('right', DIK_RIGHT)
                self._send_key_down('a', DIK_A)
                self._send_key_down('left', DIK_LEFT)
                self.last_press_time["turn_l"] = now
                self.is_turning_l = True
                self.is_turning_r = False
            actions.append(f"STEER LEFT (DNp20-L -> A + Left + Mouse dx={mouse_dx})")
        elif turn_diff < -self.turn_threshold:
            mouse_dx = gain
            self._send_mouse_move(dx=mouse_dx, dy=0)
            if not self.is_turning_r or (now - self.last_press_time["turn_r"]) > self.cooldown_duration:
                self._send_key_up('a', DIK_A)
                self._send_key_up('left', DIK_LEFT)
                self._send_key_down('d', DIK_D)
                self._send_key_down('right', DIK_RIGHT)
                self.last_press_time["turn_r"] = now
                self.is_turning_r = True
                self.is_turning_l = False
            actions.append(f"STEER RIGHT (DNp20-R -> D + Right + Mouse dx={mouse_dx})")
        else:
            if self.is_turning_l and (now - self.last_press_time["turn_l"]) > self.press_duration:
                self._send_key_up('a', DIK_A)
                self._send_key_up('left', DIK_LEFT)
                self.is_turning_l = False
            if self.is_turning_r and (now - self.last_press_time["turn_r"]) > self.press_duration:
                self._send_key_up('d', DIK_D)
                self._send_key_up('right', DIK_RIGHT)
                self.is_turning_r = False

        # 2. Locomotion Decoder: DNpe017 Forward vs MDN Moonwalker Backward (Half-Life GoldSrc: W / S)
        net_fwd = float(dnpe017_forward_rate - mdn_backward_rate)
        is_backward = (mdn_backward_rate > self.walk_threshold and mdn_backward_rate > dnpe017_forward_rate) or \
                      getattr(self, "is_unstucking", False) or is_obstacle_close

        if is_backward:
            # MDN Moonwalker backward stepping or obstacle evasion: release W and step backward on S
            if self.is_walking:
                self._send_key_up('w', DIK_W)
                self.is_walking = False
            if not self.is_stepping_back or (now - self.last_press_time.get("back", 0.0)) > self.cooldown_duration:
                self._send_key_down('s', DIK_S)
                self.last_press_time["back"] = now
                self.is_stepping_back = True
            actions.append(f"STEP BACK (MDN Moonwalker {mdn_backward_rate:.2f} -> S)")
        elif net_fwd > self.walk_threshold:
            # Clear backward step when moving forward
            if self.is_stepping_back and (now - self.last_press_time.get("back", 0.0)) > self.press_duration:
                self._send_key_up('s', DIK_S)
                self.is_stepping_back = False
            if not self.is_walking or (now - self.last_press_time["forward"]) > self.cooldown_duration:
                self._send_key_down('w', DIK_W)
                self.last_press_time["forward"] = now
                self.is_walking = True
            actions.append("WALK FORWARD (DNpe017 -> W)")
        else:
            if self.is_walking and (now - self.last_press_time["forward"]) > self.press_duration:
                self._send_key_up('w', DIK_W)
                self.is_walking = False
            if self.is_stepping_back and (now - self.last_press_time.get("back", 0.0)) > self.press_duration:
                self._send_key_up('s', DIK_S)
                self.is_stepping_back = False

        # 3. DNpe017 Attack / Fire Decoder (Half-Life GoldSrc: Left Mouse Click)
        if dnpe017_attack_rate > self.attack_threshold:
            if (now - getattr(self, "last_fire_time", 0.0)) > 0.45:
                self._send_click()
                self.last_fire_time = now
                self.last_press_time["attack"] = now
                self.is_firing = True
                actions.append("FIRE! [pydirectinput.click]")
            else:
                self.is_firing = False
        else:
            self.is_firing = False

        self.current_action = " + ".join(actions) if actions else "IDLE"

        # Recover escape keys if escape duration elapsed
        if self.is_escaping and (now - self.last_escape_time) > self.press_duration * 2:
            self._send_key_up('s', DIK_S)
            self._send_key_up('space', DIK_SPACE)
            self.is_escaping = False

        # Recover combat retaliation evasion keys (150ms backstep + strafe)
        if getattr(self, "is_retaliating", False) and (now - getattr(self, "last_retaliation_time", 0.0)) > 0.15:
            self._send_key_up('s', DIK_S)
            if hasattr(self, "retaliation_strafe_key"):
                k, sc = self.retaliation_strafe_key
                self._send_key_up(k, sc)
            self.is_retaliating = False

        # Recover obstacle unstuck keys (450ms backstep duration to actually clear wall collision)
        if getattr(self, "is_unstucking", False):
            if (now - getattr(self, "last_unstuck_time", 0.0)) > 0.45:
                self._send_key_up('s', DIK_S)
                if self.is_turning_l:
                    self._send_key_up('a', DIK_A)
                    self._send_key_up('left', DIK_LEFT)
                    self.is_turning_l = False
                if self.is_turning_r:
                    self._send_key_up('d', DIK_D)
                    self._send_key_up('right', DIK_RIGHT)
                    self.is_turning_r = False
                self.is_unstucking = False
            else:
                actions.append("UNSTUCK EVASION (S + Turn)")

        return {
            "turn_diff": float(turn_diff),
            "forward_rate": float(dnpe017_forward_rate),
            "backward_rate": float(mdn_backward_rate),
            "net_forward": float(net_fwd),
            "attack_rate": float(dnpe017_attack_rate),
            "action": self.current_action,
            "is_firing": self.is_firing,
            "is_escaping": self.is_escaping
        }

    def trigger_giant_fiber_escape(self, spin_direction: int = 1) -> Dict[str, Any]:
        """
        Executes an immediate Giant Fiber (GF) emergency escape reflex.
        Bypasses regular locomotion:
          1. 180-degree rapid mouse spin flick (dx = ±240).
          2. Backward evasive leap: keyDown('s') + keyDown('space') (jump).
          3. Releases forward 'w' key immediately.
        """
        if not self.can_send_input():
            return {"action": "SUPPRESSED (Safety Guard)", "flick_dx": 0, "keys": []}
            
        now = time.time()
        self.is_escaping = True
        self.last_escape_time = now
        
        # 1. Immediate Retaliation Fire during emergency reflex!
        self._send_click()

        # 2. Release forward movement
        self._send_key_up('w', DIK_W)
        
        # 3. 180-degree rapid mouse flick
        flick_dx = 240 * (1 if spin_direction >= 0 else -1)
        self._send_mouse_move(dx=flick_dx, dy=0)
        
        # 4. Backward evasive leap (S + Space)
        self._send_key_down('s', DIK_S)
        self._send_key_down('space', DIK_SPACE)
        
        self.last_press_time["turn_l"] = now
        self.last_press_time["turn_r"] = now
        self.last_press_time["forward"] = now
        self.last_press_time["attack"] = now
        
        logger.warning("🚨 [GIANT FIBER] Emergency escape + retaliation fire dispatched: Shot + 180° Spin + Back Leap!")
        
        return {
            "action": "GIANT FIBER ESCAPE + FIRE",
            "flick_dx": flick_dx,
            "keys": ["s", "space", "fire"]
        }

    def trigger_combat_retaliation(self) -> Dict[str, Any]:
        """
        Executes an immediate combat retaliation reflex upon taking damage:
          1. Fires primary weapon instantly (_send_click).
          2. Performs an evasive combat strafe hop (Space + A or D).
        """
        if not self.can_send_input():
            return {"action": "SUPPRESSED (Safety Guard)", "is_firing": False}

        now = time.time()
        # Cooldown guard: at most 1 retaliation burst every 0.35 seconds!
        if (now - getattr(self, "last_combat_retaliation_time", 0.0)) < 0.35:
            return {"action": self.current_action, "is_firing": False}

        self.last_combat_retaliation_time = now
        self.last_fire_time = now
        self.is_firing = True
        self.last_press_time["attack"] = now
        
        # 1. Non-blocking Retaliation Fire!
        self._send_click(duration=0.01)

        # 2. Combat Evasive Backstep + Strafe (Non-blocking: key down, auto-released asynchronously after 150ms)
        self._send_key_down('s', DIK_S)
        strafe_key = 'a' if (int(now * 10) % 2 == 0) else 'd'
        scancode = DIK_A if strafe_key == 'a' else DIK_D
        self._send_key_down(strafe_key, scancode)
        self.is_retaliating = True
        self.retaliation_strafe_key = (strafe_key, scancode)
        self.last_retaliation_time = now

        logger.warning("[RETALIATION] Can azaldi: Dusmana karsi ates acildi ve strafe yapildi!")
        try:
            print("[RETALIATION FIRE] Sinek vuruldu: Dusmana karsi ates acildi ve strafe yapildi!", flush=True)
        except Exception:
            pass
        return {
            "action": "COMBAT RETALIATION FIRE",
            "is_firing": True
        }

    def trigger_obstacle_turn(self, direction: int = 1) -> Dict[str, Any]:
        """
        Executes a swift obstacle un-stuck maneuver:
        Releases W, steps back firmly on S, and executes a full 90° - 120° Saccade
        in the escape direction (±550 dx) to route around the wall geometry.
        """
        if not self.can_send_input():
            return {"action": "SUPPRESSED (Safety Guard)", "turn_dx": 0}

        now = time.time()
        # Enforce minimum 0.4s cooldown to avoid spamming within the same panic cycle
        if (now - getattr(self, "last_unstuck_time", 0.0)) < 0.4:
            return {"action": self.current_action, "turn_dx": 0}

        self.is_unstucking = True
        self.last_unstuck_time = now
        
        # 1. Release forward walk immediately
        self._send_key_up('w', DIK_W)
        self.is_walking = False
        
        # 2. Step backward firmly (break contact with wall geometry)
        self._send_key_down('s', DIK_S)
        
        # 3. Rotate mouse away from wall (Full 90° - 120° Saccade)
        turn_dx = 550 * (1 if direction >= 0 else -1)
        self._send_mouse_move(dx=turn_dx, dy=0)
        
        # 4. Strafe and keyboard turn away from wall
        if direction >= 0:
            self._send_key_down('d', DIK_D)
            self._send_key_down('right', DIK_RIGHT)
            self.last_press_time["turn_r"] = now
            self.is_turning_r = True
            self.is_turning_l = False
        else:
            self._send_key_down('a', DIK_A)
            self._send_key_down('left', DIK_LEFT)
            self.last_press_time["turn_l"] = now
            self.is_turning_l = True
            self.is_turning_r = False

        self.last_press_time["forward"] = now + 0.55 # Lock forward key during evasion
        self.current_action = f"OBSTACLE ESCAPE (90° {'RIGHT' if direction >= 0 else 'LEFT'})"
        logger.info(f"🧱 Obstacle deadlock broken: Backed up and 90° Saccade {'RIGHT' if direction >= 0 else 'LEFT'}.")
        return {"action": self.current_action, "turn_dx": turn_dx}

    def handle_respawn(self):
        """Dispatches an auto-respawn click / spacebar if killed in Half-Life."""
        if not self.can_send_input():
            return
        self._send_click()
        self._send_key_event(DIK_SPACE, is_up=False)
        time.sleep(0.01)
        self._send_key_event(DIK_SPACE, is_up=True)

    def release_all(self):
        """Cleanly releases all hardware inputs upon termination."""
        if not self.active_keys and not self.is_walking and not self.is_turning_l and not self.is_turning_r and not self.is_stepping_back:
            return
        for key_char, scancode in [('w', DIK_W), ('a', DIK_A), ('s', DIK_S), ('d', DIK_D), ('left', DIK_LEFT), ('right', DIK_RIGHT)]:
            self._send_key_up(key_char, scancode)
        for key in [DIK_SPACE, DIK_LCONTROL]:
            self._send_key_event(key, is_up=True)
        self.active_keys.clear()
        self.is_walking = False
        self.is_turning_l = False
        self.is_turning_r = False
        self.is_stepping_back = False
        logger.info("All hardware input keys released.")
