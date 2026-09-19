"""
input_bridge.py - Descending Motor Neuron Decoder & DirectX DirectInput Coordinator
Coordinates NeuralDecoder (EMA Low-Pass Filter), LocomotionController (Smooth Pure Turn & Walk),
and ReflexManager (Anti-Stuck & Combat Saccades) with fail-safe window guards.
"""
import logging
import time
import ctypes
import sys
from typing import Dict, Any, Optional
import numpy as np

from core.platform import get_input_driver, ActionKey
from motor import NeuralDecoder, LocomotionController, ReflexManager, ReflexState

logger = logging.getLogger("FlyBrain.InputBridge")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# DirectInput Hardware Scancodes (kept as backwards-compatible aliases)
DIK_W = 0x11
DIK_A = 0x1E
DIK_S = 0x1F
DIK_D = 0x20
DIK_SPACE = 0x39
DIK_LCONTROL = 0x1D
DIK_LEFT = 0xCB
DIK_RIGHT = 0xCD
VK_F9 = 0x78
VK_F10 = 0x79
VK_F12 = 0x7B


class InputBridge:
    """
    Modular Input Coordinator.
    Orchestrates:
      - NeuralDecoder: EMA low-pass filtered biological decoding
      - LocomotionController: Smooth mouse yaw, pitch, and state-based forward walking (no strafe interference)
      - ReflexManager: Biological emergency escapes, obstacle saccades, and combat retaliation
      - Fail-safe Guards: Window focus, in-game cursor/menu detection, and F10 toggle pause
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
        self._last_step_time: float = time.time()

        # Platform Input Driver (Windows SendInput or Linux evdev/uinput)
        self.input_driver = get_input_driver(dry_run=self.dry_run)

        # DOOMFLY-Inspired Modular Subsystems
        self.decoder = NeuralDecoder(
            tau_ms=100.0,
            turn_gain=5.5,
            forward_gain=3.5,
            turn_deadzone=0.010,
            walk_threshold=0.02,
            attack_threshold=self.attack_threshold,
            decision_interval=0.08
        )
        self.locomotion = LocomotionController(
            input_driver=self.input_driver,
            mouse_gain_x=110.0,
            mouse_gain_y=40.0,
            walk_threshold=0.02,
            turn_key_threshold=0.35,
            turn_deadzone=0.010,
            fire_cooldown=0.35,
            enable_strafe=False
        )
        self.reflexes = ReflexManager(input_driver=self.input_driver)
        self._is_escaping_state: bool = False
        self._is_unstucking_state: bool = False

        logger.info(f"InputBridge initialized (Dry-Run: {self.dry_run}, Cooldown: {self.cooldown_duration*1000:.0f}ms, MouseGain: {self.mouse_turn_gain}).")
        logger.info("🔒 Safety Guard engaged: F10/F12 to Toggle Bot, F9 to Toggle Menu Lock, Desktop auto-lock active.")

    @property
    def is_walking(self) -> bool:
        return self.locomotion.is_walking

    @property
    def is_stepping_back(self) -> bool:
        return self.locomotion.is_stepping_back

    @property
    def is_firing(self) -> bool:
        return self.locomotion.last_action_desc.find("FIRE") != -1 or self.reflexes.current_state == "COMBAT_RETALIATION"

    @property
    def is_escaping(self) -> bool:
        return self.reflexes.is_active() or self._is_escaping_state

    @is_escaping.setter
    def is_escaping(self, val: bool):
        self._is_escaping_state = val

    @property
    def is_unstucking(self) -> bool:
        return self.reflexes.current_state == "OBSTACLE_SACCADE" or self._is_unstucking_state

    @is_unstucking.setter
    def is_unstucking(self, val: bool):
        self._is_unstucking_state = val

    @property
    def last_fire_time(self) -> float:
        return self.locomotion.last_fire_time

    @last_fire_time.setter
    def last_fire_time(self, val: float):
        self.locomotion.last_fire_time = val

    @property
    def current_action(self) -> str:
        if self.user_paused:
            return "PAUSED [USER_PAUSED]"
        if self.reflexes.is_active():
            return f"REFLEX: {self.reflexes.current_state}"
        return self.locomotion.last_action_desc

    def is_allowed_overlay_or_telemetry(self, fg: int) -> bool:
        """Checks if the foreground window is NVIDIA GeForce Overlay or FlyBrain Telemetry window."""
        if not fg or sys.platform != "win32":
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

            if any(k in pname for k in ['nvcontainer', 'nvidia share', 'shadowplay', 'nvidia geforce overlay']) or \
               any(k in title for k in ['nvidia', 'geforce overlay', 'shadowplay']):
                return True

            if any(k in pname for k in ['python.exe', 'pythonw.exe']) and \
               any(k in title for k in ['flybrain', 'telemetry', 'connectome']):
                return True

            return False
        except Exception:
            return False

    def is_game_focused(self) -> bool:
        """Verifies if the active foreground window is Half-Life / target game window or allowed overlay."""
        if self.dry_run or sys.platform != "win32":
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

            if self.is_allowed_overlay_or_telemetry(fg):
                return True

            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
            hProc = kernel32.OpenProcess(0x1000, False, pid.value)
            if hProc:
                buf = ctypes.create_unicode_buffer(1024)
                sz = ctypes.wintypes.DWORD(1024)
                if kernel32.QueryFullProcessImageNameW(hProc, 0, buf, ctypes.byref(sz)):
                    pname = buf.value.lower()
                    if any(k in pname for k in ['hl.exe', 'cstrike.exe', 'half-life.exe', 'cs.exe', 'hl_linux', 'left4dead2.exe', 'gta-vc.exe', 'gta_sa.exe', 'ut.exe']):
                        kernel32.CloseHandle(hProc)
                        self.target_hwnd = fg
                        return True
                kernel32.CloseHandle(hProc)

            length = user32.GetWindowTextLengthW(fg)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(fg, buff, length + 1)
                title = buff.value.lower()
                if any(k in title for k in ["half-life", "valve001", "hl.exe", "doom", "gzdoom", "counter-strike"]):
                    self.target_hwnd = fg
                    return True
            return False
        except Exception:
            return False

    def is_cursor_visible(self) -> bool:
        """Detects if the mouse cursor is showing while Half-Life is focused."""
        if self.dry_run or not self.menu_lock_enabled:
            return False
        return self.input_driver.is_cursor_visible()

    def can_send_input(self) -> bool:
        """Master safety lock. Returns True only if safe to dispatch hardware inputs."""
        if self.user_paused:
            return False
        if self.dry_run:
            return True
        if sys.platform != "win32":
            return not self.is_cursor_visible()
        try:
            fg = ctypes.windll.user32.GetForegroundWindow()
            if self.is_allowed_overlay_or_telemetry(fg):
                return False
            if not self.is_game_focused():
                return False
            if self.is_cursor_visible():
                return False
            return True
        except Exception:
            return True

    def check_hotkeys(self) -> bool:
        """Listens for F9 (Menu Lock toggle) and F10 / F12 (Bot pause toggle) on Windows."""
        if sys.platform != "win32":
            return False
        try:
            user32 = ctypes.windll.user32
            now = time.time()
            if (now - self._last_hotkey_time) < 0.35:
                return False

            state_f9 = user32.GetAsyncKeyState(VK_F9)
            if (state_f9 & 0x8000) or (state_f9 & 0x0001):
                self._last_hotkey_time = now
                self.toggle_menu_lock()
                return True

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
            logger.info("⏸️ [PAUSE] Sinek DURDURULDU (F10 / Web Stop). Girişler donduruldu.")
        else:
            self.pause_reason = ""
            logger.info("▶️ [RESUME] Sinek AKTİF. Girişler devrede.")
        return self.user_paused

    def trigger_obstacle_turn(self, direction: int = 1) -> Dict[str, Any]:
        """Triggers obstacle unstuck saccade through ReflexManager."""
        turn_dx = 550 * (1 if direction >= 0 else -1)
        self.is_unstucking = True
        if not self.can_send_input():
            return {"action": "SUPPRESSED (Safety Guard)", "turn_dx": turn_dx}
        res = self.reflexes.trigger_obstacle_saccade(direction=direction)
        side = "RIGHT" if direction >= 0 else "LEFT"
        desc = f"OBSTACLE ESCAPE ({side})"
        logger.info(f"🧱 Obstacle deadlock broken: Backed up and 90° Saccade {side}.")
        return {"action": desc, "turn_dx": turn_dx}

    def trigger_giant_fiber_escape(self) -> Dict[str, Any]:
        """Dispatches emergency Giant Fiber 180° escape reflex."""
        flick_dx = 800
        self.input_driver.mouse_move_relative(dx=flick_dx, dy=0)
        self.input_driver.press_action(ActionKey.BACKWARD)
        self.input_driver.press_action(ActionKey.JUMP)
        self.input_driver.mouse_click(duration=0.01)
        self.last_escape_time = time.time()
        self._is_escaping_state = True
        return {
            "action": "GIANT FIBER ESCAPE + FIRE",
            "flick_dx": flick_dx,
            "keys": ["s", "space", "fire"]
        }

    def trigger_combat_retaliation(self, direction: int = 1) -> Dict[str, Any]:
        """Triggers combat counter-fire, 180 whip turn, and evasion through ReflexManager."""
        if not self.can_send_input():
            return {"action": "SUPPRESSED (Safety Guard)", "is_firing": False}
        res = self.reflexes.trigger_combat_retaliation(direction=direction, flick_pixels=520)
        logger.warning(f"[RETALIATION] Can azaldı: 180° dönüş, karşı ateş açıldı ve kaçış strafe'i yapıldı! (Yön: {direction})")
        return res

    def handle_respawn(self):
        """Dispatches an auto-respawn spacebar / click if killed in Half-Life."""
        if not self.can_send_input():
            return
        self.input_driver.mouse_click(duration=0.01)
        self.input_driver.press_action(ActionKey.JUMP)
        time.sleep(0.01)
        self.input_driver.release_action(ActionKey.JUMP)

    def decode_and_dispatch(
        self,
        dnp20_left_rate: float,
        dnp20_right_rate: float,
        dnpe017_forward_rate: float,
        dnpe017_attack_rate: float,
        is_obstacle_close: bool = False,
        mdn_backward_rate: float = 0.0,
        vertical_pitch_rate: float = 0.0,
        visual_horizon: Optional[Dict[str, float]] = None,
        glance_pitch: Optional[float] = None,
        glance_duration: float = 0.4
    ) -> Dict[str, Any]:
        """
        Decodes descending neuron rates with EMA smoothing and dispatches debounced motor commands.
        """
        now = time.time()
        dt = max(0.02, min(0.1, now - self._last_step_time))
        self._last_step_time = now

        # 0. Global Safety Guards
        self.check_hotkeys()

        if self.user_paused:
            self.release_all()
            return {
                "turn_diff": 0.0, "forward_rate": 0.0, "backward_rate": 0.0, "attack_rate": 0.0,
                "action": "PAUSED [F10 / STOP BUTONU]", "is_firing": False, "is_escaping": False,
                "is_paused": True, "pause_reason": "USER_PAUSED"
            }

        if sys.platform == "win32":
            try:
                fg = ctypes.windll.user32.GetForegroundWindow()
                if self.is_allowed_overlay_or_telemetry(fg):
                    self.release_all()
                    return {
                        "turn_diff": 0.0, "forward_rate": 0.0, "backward_rate": 0.0, "attack_rate": 0.0,
                        "action": "OBSERVING (OVERLAY / TELEMETRY ODAKTA)", "is_firing": False, "is_escaping": False,
                        "is_paused": False, "pause_reason": ""
                    }
            except Exception:
                pass

        if not self.is_game_focused():
            self.release_all()
            return {
                "turn_diff": 0.0, "forward_rate": 0.0, "backward_rate": 0.0, "attack_rate": 0.0,
                "action": "PAUSED [ODAK YOK - MASAÜSTÜ]", "is_firing": False, "is_escaping": False,
                "is_paused": True, "pause_reason": "WINDOW_UNFOCUSED"
            }

        if self.is_cursor_visible():
            self.release_all()
            return {
                "turn_diff": 0.0, "forward_rate": 0.0, "backward_rate": 0.0, "attack_rate": 0.0,
                "action": "PAUSED [MENÜ / KONSOL AÇIK (~, ESC)]", "is_firing": False, "is_escaping": False,
                "is_paused": True, "pause_reason": "MENU_ACTIVE"
            }

        # 1. Update Emergency Reflex FSM (Obstacle Saccade / Retaliation override)
        if self.reflexes.update(now=now):
            return {
                "turn_diff": 0.0,
                "forward_rate": float(dnpe017_forward_rate),
                "backward_rate": float(mdn_backward_rate),
                "attack_rate": float(dnpe017_attack_rate),
                "action": f"EMERGENCY REFLEX ({self.reflexes.current_state})",
                "is_firing": self.reflexes.current_state == ReflexState.COMBAT_RETALIATION or getattr(self.reflexes.current_state, "name", "") == "COMBAT_RETALIATION",
                "is_escaping": True,
                "is_paused": False,
                "pause_reason": ""
            }

        # 2. DOOMFLY-Inspired Neural Decoding with EMA Low-Pass Filter
        decoded = self.decoder.decode(
            dnp20_left_rate=dnp20_left_rate,
            dnp20_right_rate=dnp20_right_rate,
            dnpe017_forward_rate=dnpe017_forward_rate,
            dnpe017_attack_rate=dnpe017_attack_rate,
            mdn_backward_rate=mdn_backward_rate,
            vertical_pitch_rate=vertical_pitch_rate,
            dt=dt,
            now=now
        )

        # 3. Smooth Locomotion & Steering Execution
        if visual_horizon:
            decoded["visual_horizon"] = visual_horizon
        if glance_pitch is not None:
            decoded["glance_pitch"] = glance_pitch
            decoded["glance_duration"] = glance_duration

        suppress = (not self.dry_run and not self.can_send_input())
        loco_info = self.locomotion.apply(decoded, now=now, suppress_input=suppress)

        if loco_info.get("is_firing", False):
            logger.info(f"🔥 [FIRE] Primary Attack Dispatched: smooth_rate={self.decoder.smooth_attack:.3f}")

        return {
            "turn_diff": decoded["raw_turn_diff"],
            "forward_rate": float(dnpe017_forward_rate),
            "backward_rate": float(mdn_backward_rate),
            "net_forward": decoded["net_forward"],
            "attack_rate": float(dnpe017_attack_rate),
            "action": loco_info.get("action", "IDLE"),
            "is_firing": loco_info.get("is_firing", False),
            "is_escaping": False,
            "is_paused": False,
            "pause_reason": "",
            "haltere": loco_info.get("haltere", {})
        }

    def request_glance(self, target_pitch: float, duration: float = 0.4) -> None:
        """Forwards transient glance request to locomotion haltere."""
        self.locomotion.request_glance(target_pitch=target_pitch, duration=duration)

    def release_all(self):
        """Failsafe release of all hardware inputs and resets subsystems."""
        had_active = bool(self.input_driver.active_actions or self.locomotion.is_walking or self.locomotion.is_stepping_back)
        self.reflexes.release_all()
        self.locomotion.release_all()
        self.input_driver.release_all()
        if had_active:
            logger.info("All hardware input keys released.")
