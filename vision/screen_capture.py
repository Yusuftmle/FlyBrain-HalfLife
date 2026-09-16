"""
Screen Capture Module (Compatibility Layer)
Delegates to core.platform HAL drivers for cross-platform low-latency frame capture.
"""
import numpy as np
from typing import Optional
from core.platform import get_screen_capture_driver


class ScreenCapture:
    """
    Captures Half-Life or target game window at 60+ FPS via core.platform HAL.
    """
    def __init__(self, window_title: str = "Half-Life"):
        self.window_title = window_title
        self.driver = get_screen_capture_driver(target_title=window_title)

    @property
    def hwnd(self) -> Optional[int]:
        return getattr(self.driver, "hwnd", None)

    @property
    def sct(self):
        return getattr(self.driver, "sct", None)

    def grab_frame(self) -> np.ndarray:
        """Grabs active game frame and returns BGR image."""
        frame = self.driver.grab_frame()
        if frame is not None and frame.size > 0:
            return frame
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        self.driver.close()
