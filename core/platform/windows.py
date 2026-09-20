"""
windows.py - Windows Platform Drivers for Screen Capture and Input Injection
Uses Win32 GDI / PrintWindow and DirectX SendInput for sub-millisecond execution.
"""
import atexit
import ctypes
import ctypes.wintypes
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import mss
import numpy as np

from .base import BaseScreenCapture, BaseInputDriver, ActionKey

logger = logging.getLogger("FlyBrain.Platform.Windows")

# --- Windows C Structure Definitions for Input Injection ---
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

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("hCursor", ctypes.c_void_p),
        ("ptScreenPos", ctypes.wintypes.POINT)
    ]

class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ('dwSize', ctypes.wintypes.DWORD),
        ('cntUsage', ctypes.wintypes.DWORD),
        ('th32ProcessID', ctypes.wintypes.DWORD),
        ('th32DefaultHeapID', ctypes.POINTER(ctypes.c_ulong)),
        ('th32ModuleID', ctypes.wintypes.DWORD),
        ('cntThreads', ctypes.wintypes.DWORD),
        ('th32ParentProcessID', ctypes.wintypes.DWORD),
        ('pcPriClassBase', ctypes.c_long),
        ('dwFlags', ctypes.wintypes.DWORD),
        ('szExeFile', ctypes.c_wchar * 260)
    ]

# Win32 Constants
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
CURSOR_SHOWING = 0x00000001

# DirectInput Hardware Scancodes
WIN_SCANCODES = {
    ActionKey.FORWARD: 0x11,        # DIK_W
    ActionKey.BACKWARD: 0x1F,       # DIK_S
    ActionKey.STRAFE_LEFT: 0x1E,    # DIK_A
    ActionKey.STRAFE_RIGHT: 0x20,   # DIK_D
    ActionKey.TURN_LEFT: 0xCB,      # DIK_LEFT
    ActionKey.TURN_RIGHT: 0xCD,     # DIK_RIGHT
    ActionKey.JUMP: 0x39,           # DIK_SPACE
    ActionKey.CROUCH: 0x1D,         # DIK_LCONTROL
    ActionKey.QUICKLOAD: 0x78       # VK_F9
}


class WindowsScreenCapture(BaseScreenCapture):
    """
    Windows Low-Latency Game Window Frame Grabber.
    Uses cached Win32 GDI / PrintWindow (PW_RENDERFULLCONTENT) as primary path,
    with mss as low-latency fallback.
    """
    def __init__(self, target_title: str = "Half-Life"):
        self.target_title = target_title
        self.hwnd: Optional[int] = None
        self.sct: mss.MSS = mss.MSS()

        # GDI persistent context caches (avoids per-frame allocations)
        self._gdi_hwnd: Optional[int] = None
        self._gdi_hdc_window: Optional[int] = None
        self._gdi_hdc_mem: Optional[int] = None
        self._gdi_hbm: Optional[int] = None
        self._gdi_old_bm: Optional[int] = None
        self._gdi_w: int = 0
        self._gdi_h: int = 0
        self._gdi_buf: Optional[np.ndarray] = None
        self._gdi_bih: Any = None

        self.find_target_window()

    def _cleanup_gdi(self):
        """Releases cached Win32 GDI contexts cleanly."""
        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            if self._gdi_hdc_mem:
                if self._gdi_old_bm:
                    gdi32.SelectObject(self._gdi_hdc_mem, self._gdi_old_bm)
                if self._gdi_hbm:
                    gdi32.DeleteObject(self._gdi_hbm)
                gdi32.DeleteDC(self._gdi_hdc_mem)
            if self._gdi_hdc_window and self._gdi_hwnd:
                user32.ReleaseDC(self._gdi_hwnd, self._gdi_hdc_window)
        except Exception:
            pass
        self._gdi_hwnd = None
        self._gdi_hdc_window = None
        self._gdi_hdc_mem = None
        self._gdi_hbm = None
        self._gdi_old_bm = None
        self._gdi_w = 0
        self._gdi_h = 0
        self._gdi_buf = None
        self._gdi_bih = None

    def find_target_window(self, titles: Optional[List[str]] = None, process_names: Optional[List[str]] = None) -> bool:
        """Locates game window handle using Win32 API FindWindow and Process Enumeration."""
        if titles is None:
            titles = [self.target_title, "Half-Life", "hl", "Counter-Strike", "DOOM", "GZDoom"]
        if process_names is None:
            process_names = ['hl.exe', 'cstrike.exe', 'half-life.exe', 'cs.exe', 'hl_linux']

        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            # Connect to interactive desktop if needed
            try:
                hinput = user32.OpenInputDesktop(0, False, 0x01FF)
                if hinput:
                    user32.SetThreadDesktop(hinput)
            except Exception:
                pass

            # 1. Direct Search by Title
            for title in titles:
                hwnd = user32.FindWindowW(None, title)
                if hwnd != 0 and user32.IsWindowVisible(hwnd):
                    self.hwnd = hwnd
                    logger.info(f"WindowsCapture: Attached to game window by title '{title}' (HWND: {hwnd})")
                    return True

            # 2. Search by GoldSrc and Steam SDL2 window class
            for cls_name in ["SDL_app", "Valve001"]:
                for title in titles + [None]:
                    hwnd = user32.FindWindowW(cls_name, title)
                    if hwnd != 0 and user32.IsWindowVisible(hwnd):
                        pid = ctypes.wintypes.DWORD()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        hProc = kernel32.OpenProcess(0x1000, False, pid.value)
                        if hProc:
                            buf = ctypes.create_unicode_buffer(1024)
                            sz = ctypes.wintypes.DWORD(1024)
                            if kernel32.QueryFullProcessImageNameW(hProc, 0, buf, ctypes.byref(sz)):
                                pname = buf.value.lower()
                                if any(k in pname for k in process_names):
                                    kernel32.CloseHandle(hProc)
                                    self.hwnd = hwnd
                                    logger.info(f"WindowsCapture: Attached via class '{cls_name}': HWND {hwnd}")
                                    return True
                            kernel32.CloseHandle(hProc)

            # 3. Direct Enumeration of visible windows matching process names
            found = None
            def enum_cb(h, _):
                nonlocal found
                if user32.IsWindowVisible(h):
                    pid = ctypes.wintypes.DWORD()
                    user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
                    hProc = kernel32.OpenProcess(0x1000, False, pid.value)
                    if hProc:
                        buf = ctypes.create_unicode_buffer(1024)
                        sz = ctypes.wintypes.DWORD(1024)
                        if kernel32.QueryFullProcessImageNameW(hProc, 0, buf, ctypes.byref(sz)):
                            pname = buf.value.lower()
                            if any(k in pname for k in process_names):
                                rect = ctypes.wintypes.RECT()
                                user32.GetWindowRect(h, ctypes.byref(rect))
                                if (rect.right - rect.left) >= 100 and (rect.bottom - rect.top) >= 100:
                                    found = h
                                    kernel32.CloseHandle(hProc)
                                    return False
                        kernel32.CloseHandle(hProc)
                return True

            WND = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
            user32.EnumWindows(WND(enum_cb), 0)
            if found:
                self.hwnd = found
                logger.info(f"WindowsCapture: Attached via process scan (HWND: {found})")
                return True

        except Exception as e:
            logger.error(f"WindowsCapture: Error finding game window: {e}")

        logger.info(f"WindowsCapture: Window '{self.target_title}' not directly bound. Using process/monitor fallback.")
        return False

    def _capture_via_printwindow(self) -> Optional[np.ndarray]:
        """Captures game window buffer using Win32 PrintWindow (works even when occluded)."""
        if not self.hwnd:
            return None
        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            rect = ctypes.wintypes.RECT()
            user32.GetClientRect(self.hwnd, ctypes.byref(rect))
            w = int(rect.right - rect.left)
            h = int(rect.bottom - rect.top)
            if w < 100 or h < 100:
                return None

            if (self._gdi_hwnd != self.hwnd or self._gdi_w != w or self._gdi_h != h or 
                self._gdi_hdc_mem is None or self._gdi_hbm is None):
                self._cleanup_gdi()
                hdc_window = user32.GetDC(self.hwnd)
                if not hdc_window:
                    return None
                hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
                hbm = gdi32.CreateCompatibleBitmap(hdc_window, w, h)
                old_bm = gdi32.SelectObject(hdc_mem, hbm)

                class BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ('biSize', ctypes.wintypes.DWORD),
                        ('biWidth', ctypes.wintypes.LONG),
                        ('biHeight', ctypes.wintypes.LONG),
                        ('biPlanes', ctypes.wintypes.WORD),
                        ('biBitCount', ctypes.wintypes.WORD),
                        ('biCompression', ctypes.wintypes.DWORD),
                        ('biSizeImage', ctypes.wintypes.DWORD),
                        ('biXPelsPerMeter', ctypes.wintypes.LONG),
                        ('biYPelsPerMeter', ctypes.wintypes.LONG),
                        ('biClrUsed', ctypes.wintypes.DWORD),
                        ('biClrImportant', ctypes.wintypes.DWORD)
                    ]
                bih = BITMAPINFOHEADER()
                bih.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bih.biWidth = w
                bih.biHeight = -h # top-down DIB
                bih.biPlanes = 1
                bih.biBitCount = 32
                bih.biCompression = 0

                self._gdi_hwnd = self.hwnd
                self._gdi_hdc_window = hdc_window
                self._gdi_hdc_mem = hdc_mem
                self._gdi_hbm = hbm
                self._gdi_old_bm = old_bm
                self._gdi_w = w
                self._gdi_h = h
                self._gdi_bih = bih
                self._gdi_buf = np.empty((h, w, 4), dtype=np.uint8)

            res = user32.PrintWindow(self.hwnd, self._gdi_hdc_mem, 2) # PW_RENDERFULLCONTENT
            if not res:
                res = user32.PrintWindow(self.hwnd, self._gdi_hdc_mem, 0)

            frame_bgr = None
            if res:
                gdi32.GetDIBits(
                    self._gdi_hdc_mem, self._gdi_hbm, 0, h, 
                    self._gdi_buf.ctypes.data_as(ctypes.c_void_p), 
                    ctypes.byref(self._gdi_bih), 0
                )
                frame_bgr = cv2.cvtColor(self._gdi_buf, cv2.COLOR_BGRA2BGR)

            if frame_bgr is not None and frame_bgr.size > 0 and float(np.mean(frame_bgr)) > 0.5:
                return frame_bgr
            return None
        except Exception:
            self._cleanup_gdi()
            return None

    def grab_frame(self) -> Optional[np.ndarray]:
        """Grabs active game screen via PrintWindow (primary, captures 3D OpenGL/DWM buffer) or MSS (fallback)."""
        # 1. Primary: Direct Window-Buffer Capture via PrintWindow
        if self.hwnd:
            frame = self._capture_via_printwindow()
            if frame is not None and frame.size > 0 and float(np.mean(frame)) > 0.5:
                return frame

        # 2. Fallback: Bound window rect via MSS (only if not black)
        if self.hwnd:
            try:
                rect = ctypes.wintypes.RECT()
                ctypes.windll.user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
                w = int(rect.right - rect.left)
                h = int(rect.bottom - rect.top)
                if w >= 100 and h >= 100 and rect.left > -1000 and rect.top > -1000:
                    monitor = {
                        "left": max(0, int(rect.left)),
                        "top": max(0, int(rect.top)),
                        "width": w,
                        "height": h
                    }
                    raw_sct = self.sct.grab(monitor)
                    bgra = np.frombuffer(raw_sct.raw, dtype=np.uint8).reshape((h, w, 4))
                    frame_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                    if float(np.mean(frame_bgr)) > 0.5:
                        return frame_bgr
            except Exception:
                pass

        # 3. Secondary Fallback: Fullscreen center crop on primary monitor
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
                raw_sct = self.sct.grab(monitor)
                bgra = np.array(raw_sct, dtype=np.uint8)
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            except Exception:
                pass

        return None

    def is_game_running(self) -> bool:
        """Returns True if the game window is currently running, or hl.exe process is active."""
        if self.hwnd and ctypes.windll.user32.IsWindow(self.hwnd):
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
            if (rect.right - rect.left) >= 50 and (rect.bottom - rect.top) >= 50:
                return True
        self.find_target_window()
        return self.hwnd is not None or self.is_game_process_running()

    def is_game_process_running(self) -> bool:
        """Checks if hl.exe or target game process is active on the system."""
        try:
            kernel32 = ctypes.windll.kernel32
            hSnap = kernel32.CreateToolhelp32Snapshot(0x00000002, 0) # TH32CS_SNAPPROCESS
            if hSnap == -1:
                return False
            pe = PROCESSENTRY32W()
            pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            found = False
            targets = ['hl.exe', 'cstrike.exe', 'half-life.exe', 'cs.exe', 'hl_linux']
            if kernel32.Process32FirstW(hSnap, ctypes.byref(pe)):
                while True:
                    if pe.szExeFile.lower() in targets:
                        found = True
                        break
                    if not kernel32.Process32NextW(hSnap, ctypes.byref(pe)):
                        break
            kernel32.CloseHandle(hSnap)
            return found
        except Exception:
            return False

    def focus_game_window(self) -> bool:
        """Brings game window to foreground with thread input attachment."""
        if not self.hwnd:
            self.find_target_window()
        if self.hwnd:
            try:
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32
                fg = user32.GetForegroundWindow()
                cur_tid = kernel32.GetCurrentThreadId()
                fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
                if fg_tid and cur_tid != fg_tid:
                    user32.AttachThreadInput(cur_tid, fg_tid, True)
                user32.ShowWindow(self.hwnd, 9) # SW_RESTORE
                user32.SetForegroundWindow(self.hwnd)
                if fg_tid and cur_tid != fg_tid:
                    user32.AttachThreadInput(cur_tid, fg_tid, False)
                return True
            except Exception:
                pass
        return False

    def is_game_focused(self) -> bool:
        """Returns True if the game window is the active foreground window."""
        if not self.is_game_running():
            return False
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            fg = user32.GetForegroundWindow()
            if not fg:
                return False
            if self.hwnd and fg == self.hwnd:
                return True
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
            hProc = kernel32.OpenProcess(0x1000, False, pid.value)
            if hProc:
                buf = ctypes.create_unicode_buffer(1024)
                sz = ctypes.wintypes.DWORD(1024)
                if kernel32.QueryFullProcessImageNameW(hProc, 0, buf, ctypes.byref(sz)):
                    pname = buf.value.lower()
                    if any(k in pname for k in ['hl.exe', 'cstrike.exe', 'half-life.exe']):
                        kernel32.CloseHandle(hProc)
                        self.hwnd = fg
                        return True
                kernel32.CloseHandle(hProc)
            length = user32.GetWindowTextLengthW(fg)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(fg, buff, length + 1)
                title = buff.value.lower()
                if any(k in title for k in ['half-life', 'valve001']):
                    self.hwnd = fg
                    return True
        except Exception:
            pass
        return False

    def close(self):
        self._cleanup_gdi()
        try:
            self.sct.close()
        except Exception:
            pass


class WindowsInputDriver(BaseInputDriver):
    """
    Windows DirectInput Hardware Injection Driver.
    Issues genuine scancodes via ctypes.windll.user32.SendInput.
    """
    def __init__(self, dry_run: bool = False):
        super().__init__(dry_run=dry_run)
        atexit.register(self.release_all)

    def press_action(self, action: ActionKey):
        """Holds down hardware key scancode."""
        self.active_actions.add(action)
        if self.dry_run:
            return

        scancode = WIN_SCANCODES.get(action)
        if scancode is None:
            return

        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, scancode, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception:
            pass

    def release_action(self, action: ActionKey):
        """Releases hardware key scancode."""
        self.active_actions.discard(action)
        if self.dry_run:
            return

        scancode = WIN_SCANCODES.get(action)
        if scancode is None:
            return

        try:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, scancode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(1), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        except Exception:
            pass

    def mouse_click(self, duration: float = 0.05):
        """Sends a complete primary mouse click (Fire weapon) via Win32 SendInput."""
        if self.dry_run:
            return
        extra = ctypes.c_ulong(0)
        try:
            # Mouse button down
            down = Input(
                ctypes.c_ulong(0),
                Input_I(mi=MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra)))
            )
            sent = ctypes.windll.user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(Input))
            if sent != 1:
                raise ctypes.WinError()

            if duration > 0:
                time.sleep(duration)

            # Mouse button up
            up = Input(
                ctypes.c_ulong(0),
                Input_I(mi=MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra)))
            )
            sent = ctypes.windll.user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(Input))
            if sent != 1:
                raise ctypes.WinError()
        except Exception as e:
            logger.error(f"WindowsInputDriver.mouse_click failed: {e}")

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
        """Failsafe: releases all held keys and mouse buttons."""
        for action in list(self.active_actions):
            self.release_action(action)
        self.active_actions.clear()
        if not self.dry_run:
            try:
                extra = ctypes.c_ulong(0)
                up = Input(
                    ctypes.c_ulong(0),
                    Input_I(mi=MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra)))
                )
                ctypes.windll.user32.SendInput(1, ctypes.pointer(up), ctypes.sizeof(up))
            except Exception:
                pass

    def is_cursor_visible(self) -> bool:
        """Returns True if Windows mouse cursor is visible."""
        try:
            ci = CURSORINFO()
            ci.cbSize = ctypes.sizeof(CURSORINFO)
            if ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):
                return bool(ci.flags & CURSOR_SHOWING)
        except Exception:
            pass
        return False
