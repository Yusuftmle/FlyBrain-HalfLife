"""
linux.py - Linux Platform Drivers for Screen Capture and Input Injection
Uses native MSS (X11/XWayland) and evdev / uinput (with pynput fallback).
"""
import atexit
import logging
import os
import shutil
import subprocess
import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import mss
import numpy as np

from .base import BaseScreenCapture, BaseInputDriver, ActionKey

logger = logging.getLogger("FlyBrain.Platform.Linux")

# Optional imports for Linux backends
try:
    import evdev
    from evdev import UInput, ecodes
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False

try:
    from pynput.keyboard import Controller as KeyboardController, Key
    from pynput.mouse import Controller as MouseController, Button as MouseButton
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class LinuxScreenCapture(BaseScreenCapture):
    """
    Linux Game Window Frame Grabber.
    Uses MSS for native X11/XShm direct frame grab (<2ms) and xdotool/wmctrl for geometry.
    """
    def __init__(self, target_title: str = "Half-Life"):
        self.target_title = target_title
        self.window_id: Optional[str] = None
        self.window_rect: Optional[Dict[str, int]] = None
        try:
            self.sct: Optional[mss.MSS] = mss.MSS()
        except Exception as e:
            logger.warning(f"LinuxCapture: Could not initialize MSS display connection: {e}")
            self.sct = None

        # Check for Wayland session
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            logger.info("LinuxCapture: Wayland session detected. For lowest latency, running under X11/XWayland or gamescope is recommended.")

        self.find_target_window()

    def find_target_window(self, titles: Optional[List[str]] = None, process_names: Optional[List[str]] = None) -> bool:
        """Locates game window ID using xdotool or wmctrl."""
        if titles is None:
            titles = [self.target_title, "Half-Life", "hl", "Counter-Strike"]

        # 1. Try xdotool
        if shutil.which("xdotool"):
            for title in titles:
                try:
                    res = subprocess.run(
                        ["xdotool", "search", "--onlyvisible", "--name", title],
                        capture_output=True, text=True, timeout=1.0
                    )
                    ids = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
                    if ids:
                        self.window_id = ids[0]
                        self._update_window_geometry()
                        logger.info(f"LinuxCapture: Attached to game window '{title}' via xdotool (ID: {self.window_id})")
                        return True
                except Exception:
                    pass

        # 2. Try wmctrl
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-G"], capture_output=True, text=True, timeout=1.0)
                for line in res.stdout.splitlines():
                    parts = line.split(None, 8)
                    if len(parts) >= 8:
                        wid, _, x, y, w, h, _, title = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
                        if any(t.lower() in title.lower() for t in titles):
                            self.window_id = wid
                            self.window_rect = {
                                "left": max(0, int(x)),
                                "top": max(0, int(y)),
                                "width": int(w),
                                "height": int(h)
                            }
                            logger.info(f"LinuxCapture: Attached via wmctrl '{title}' (ID: {wid})")
                            return True
            except Exception:
                pass

        logger.info(f"LinuxCapture: Target window '{self.target_title}' not directly bound by ID. Using active monitor crop.")
        return False

    def _update_window_geometry(self):
        """Fetches window bounding box using xdotool getwindowgeometry."""
        if not self.window_id or not shutil.which("xdotool"):
            return
        try:
            res = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", self.window_id],
                capture_output=True, text=True, timeout=1.0
            )
            data = {}
            for line in res.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = int(v.strip())
            if "WIDTH" in data and "HEIGHT" in data:
                self.window_rect = {
                    "left": max(0, data.get("X", 0)),
                    "top": max(0, data.get("Y", 0)),
                    "width": data["WIDTH"],
                    "height": data["HEIGHT"]
                }
        except Exception:
            pass

    def grab_frame(self) -> Optional[np.ndarray]:
        """Captures active game frame with MSS."""
        if self.sct is None:
            return None

        # 1. Target Window Capture if bounds known
        if self.window_rect and self.window_rect["width"] >= 100 and self.window_rect["height"] >= 100:
            try:
                raw = self.sct.grab(self.window_rect)
                bgra = np.array(raw, dtype=np.uint8)
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            except Exception:
                pass

        # 2. Fallback: Center crop on primary monitor
        if self.is_game_running():
            try:
                mon = self.sct.monitors[1]
                cx, cy = mon["width"] // 2, mon["height"] // 2
                crop_w, crop_h = min(800, mon["width"]), min(600, mon["height"])
                monitor = {
                    "left": cx - crop_w // 2,
                    "top": cy - crop_h // 2,
                    "width": crop_w,
                    "height": crop_h
                }
                raw = self.sct.grab(monitor)
                bgra = np.array(raw, dtype=np.uint8)
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            except Exception:
                pass

        return None

    def is_game_running(self) -> bool:
        """Checks if Half-Life window or process is alive."""
        if self.is_game_process_running():
            return True
        if self.window_id and shutil.which("xdotool"):
            try:
                res = subprocess.run(
                    ["xdotool", "getwindowname", self.window_id],
                    capture_output=True, text=True, timeout=0.5
                )
                if res.returncode == 0 and res.stdout.strip():
                    return True
            except Exception:
                pass
        return self.find_target_window()

    def is_game_process_running(self) -> bool:
        """Scans /proc for hl_linux, hl_linux-bin, or hl.exe."""
        targets = {'hl_linux', 'hl_linux-bin', 'hl.exe', 'half-life.exe', 'cstrike.exe'}
        try:
            # Check via /proc for maximum speed without spawning ps subshell
            for pid in os.listdir('/proc'):
                if pid.isdigit():
                    try:
                        with open(f"/proc/{pid}/comm", "r") as f:
                            comm = f.read().strip().lower()
                            if comm in targets:
                                return True
                    except (IOError, PermissionError):
                        continue
        except Exception:
            pass

        # Fallback to pgrep if /proc direct scan failed
        if shutil.which("pgrep"):
            for t in targets:
                try:
                    res = subprocess.run(["pgrep", "-x", t], capture_output=True, timeout=0.5)
                    if res.returncode == 0:
                        return True
                except Exception:
                    pass
        return False

    def focus_game_window(self) -> bool:
        """Brings game window to focus using xdotool or wmctrl."""
        if self.window_id and shutil.which("xdotool"):
            try:
                subprocess.run(["xdotool", "windowactivate", self.window_id], timeout=0.5)
                return True
            except Exception:
                pass
        if self.window_id and shutil.which("wmctrl"):
            try:
                subprocess.run(["wmctrl", "-ia", self.window_id], timeout=0.5)
                return True
            except Exception:
                pass
        return False

    def is_game_focused(self) -> bool:
        """Returns True if the target game window is active in the foreground."""
        if shutil.which("xdotool") and self.window_id:
            try:
                res = subprocess.run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=0.5)
                active_id = res.stdout.strip()
                return bool(active_id and (int(active_id) == int(self.window_id)))
            except Exception:
                pass
        return self.is_game_running()

    def close(self):
        try:
            if self.sct:
                self.sct.close()
        except Exception:
            pass


class LinuxInputDriver(BaseInputDriver):
    """
    Linux Two-Tier Hardware Input Driver.
    - Tier 1 (Kernel): python-evdev / uinput (zero latency virtual hardware device).
    - Tier 2 (User-Space Fallback): pynput / X11 (runs without root/uinput permissions).
    """
    def __init__(self, dry_run: bool = False):
        super().__init__(dry_run=dry_run)
        self.uinput: Optional[Any] = None
        self.pynput_kbd: Optional[Any] = None
        self.pynput_mouse: Optional[Any] = None
        self.backend: str = "dry_run"

        if not self.dry_run:
            self._init_backend()

        atexit.register(self.release_all)

    def _init_backend(self):
        """Initializes evdev / uinput or falls back to pynput."""
        # Tier 1: Try evdev / uinput
        if EVDEV_AVAILABLE:
            try:
                cap = {
                    ecodes.EV_KEY: [
                        ecodes.KEY_W, ecodes.KEY_S, ecodes.KEY_A, ecodes.KEY_D,
                        ecodes.KEY_SPACE, ecodes.KEY_LEFTCTRL, ecodes.KEY_LEFT, ecodes.KEY_RIGHT,
                        ecodes.KEY_F9, ecodes.BTN_LEFT, ecodes.BTN_RIGHT
                    ],
                    ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y]
                }
                self.uinput = UInput(cap, name="FlyBrain-Virtual-Device")
                self.backend = "evdev"
                logger.info("LinuxInput: Tier 1 (evdev/uinput) kernel virtual input device initialized.")
                return
            except PermissionError:
                logger.warning("LinuxInput: /dev/uinput PermissionError. (Run: sudo usermod -a -G input $USER). Falling back to pynput.")
            except Exception as e:
                logger.warning(f"LinuxInput: evdev initialization failed: {e}. Falling back to pynput.")

        # Tier 2: Try pynput (X11 user-space)
        if PYNPUT_AVAILABLE:
            try:
                self.pynput_kbd = KeyboardController()
                self.pynput_mouse = MouseController()
                self.backend = "pynput"
                logger.info("LinuxInput: Tier 2 (pynput/X11) user-space driver initialized.")
                return
            except Exception as e:
                logger.warning(f"LinuxInput: pynput initialization failed: {e}")

        self.backend = "simulation"
        logger.info("LinuxInput: Running in simulated input mode (physical keys disabled).")

    def press_action(self, action: ActionKey):
        """Holds down key for the semantic action."""
        self.active_actions.add(action)
        if self.dry_run or self.backend == "simulation":
            return

        if self.backend == "evdev" and self.uinput:
            ev_key = self._action_to_evdev(action)
            if ev_key:
                self.uinput.write(ecodes.EV_KEY, ev_key, 1) # KeyDown
                self.uinput.syn()
        elif self.backend == "pynput" and self.pynput_kbd:
            pk = self._action_to_pynput(action)
            if pk:
                try:
                    self.pynput_kbd.press(pk)
                except Exception:
                    pass

    def release_action(self, action: ActionKey):
        """Releases key for the semantic action."""
        self.active_actions.discard(action)
        if self.dry_run or self.backend == "simulation":
            return

        if self.backend == "evdev" and self.uinput:
            ev_key = self._action_to_evdev(action)
            if ev_key:
                self.uinput.write(ecodes.EV_KEY, ev_key, 0) # KeyUp
                self.uinput.syn()
        elif self.backend == "pynput" and self.pynput_kbd:
            pk = self._action_to_pynput(action)
            if pk:
                try:
                    self.pynput_kbd.release(pk)
                except Exception:
                    pass

    def mouse_click(self, duration: float = 0.05):
        """Sends primary mouse click (Fire weapon)."""
        if self.dry_run or self.backend == "simulation":
            return

        if self.backend == "evdev" and self.uinput:
            self.uinput.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 1)
            self.uinput.syn()
            time.sleep(duration)
            self.uinput.write(ecodes.EV_KEY, ecodes.BTN_LEFT, 0)
            self.uinput.syn()
        elif self.backend == "pynput" and self.pynput_mouse:
            try:
                self.pynput_mouse.press(MouseButton.left)
                time.sleep(duration)
                self.pynput_mouse.release(MouseButton.left)
            except Exception:
                pass

    def mouse_move_relative(self, dx: int, dy: int):
        """Moves mouse relative delta (Yaw/Pitch view rotation)."""
        if self.dry_run or self.backend == "simulation":
            return

        if self.backend == "evdev" and self.uinput:
            self.uinput.write(ecodes.EV_REL, ecodes.REL_X, dx)
            self.uinput.write(ecodes.EV_REL, ecodes.REL_Y, dy)
            self.uinput.syn()
        elif self.backend == "pynput" and self.pynput_mouse:
            try:
                self.pynput_mouse.move(dx, dy)
            except Exception:
                pass

    def release_all(self):
        """Emergency failsafe: releases all held keys."""
        for action in list(self.active_actions):
            self.release_action(action)
        self.active_actions.clear()

        if self.uinput:
            try:
                self.uinput.close()
            except Exception:
                pass
            self.uinput = None

    def is_cursor_visible(self) -> bool:
        """On Linux, cursor visibility inspection is typically handled via XFixes or gamescope."""
        return False

    def _action_to_evdev(self, action: ActionKey) -> Optional[int]:
        """Maps semantic action to Linux evdev scancode."""
        if not EVDEV_AVAILABLE:
            return None
        mapping = {
            ActionKey.FORWARD: ecodes.KEY_W,
            ActionKey.BACKWARD: ecodes.KEY_S,
            ActionKey.STRAFE_LEFT: ecodes.KEY_A,
            ActionKey.STRAFE_RIGHT: ecodes.KEY_D,
            ActionKey.TURN_LEFT: ecodes.KEY_LEFT,
            ActionKey.TURN_RIGHT: ecodes.KEY_RIGHT,
            ActionKey.JUMP: ecodes.KEY_SPACE,
            ActionKey.CROUCH: ecodes.KEY_LEFTCTRL,
            ActionKey.QUICKLOAD: ecodes.KEY_F9
        }
        return mapping.get(action)

    def _action_to_pynput(self, action: ActionKey):
        """Maps semantic action to pynput Key or character."""
        if not PYNPUT_AVAILABLE:
            return None
        mapping = {
            ActionKey.FORWARD: 'w',
            ActionKey.BACKWARD: 's',
            ActionKey.STRAFE_LEFT: 'a',
            ActionKey.STRAFE_RIGHT: 'd',
            ActionKey.TURN_LEFT: Key.left,
            ActionKey.TURN_RIGHT: Key.right,
            ActionKey.JUMP: Key.space,
            ActionKey.CROUCH: Key.ctrl_l,
            ActionKey.QUICKLOAD: Key.f9
        }
        return mapping.get(action)
