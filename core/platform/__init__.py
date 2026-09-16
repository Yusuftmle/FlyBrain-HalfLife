"""
core.platform - Hardware Abstraction Layer (HAL)
Dynamic factory loading platform-specific screen capture and hardware input drivers.
"""
import os
import sys
import logging
from typing import Tuple

from .base import BaseScreenCapture, BaseInputDriver, ActionKey

logger = logging.getLogger("FlyBrain.Platform")


def get_screen_capture_driver(target_title: str = "Half-Life") -> BaseScreenCapture:
    """Dynamically returns the appropriate screen capture driver for the current OS."""
    override = os.environ.get("FLYBRAIN_PLATFORM", "").lower()

    if override == "mock":
        from .mock import MockScreenCapture
        return MockScreenCapture(target_title=target_title)

    if sys.platform == "win32":
        from .windows import WindowsScreenCapture
        return WindowsScreenCapture(target_title=target_title)
    elif sys.platform.startswith("linux"):
        from .linux import LinuxScreenCapture
        return LinuxScreenCapture(target_title=target_title)
    else:
        logger.warning(f"Unsupported OS '{sys.platform}'. Defaulting to MockScreenCapture.")
        from .mock import MockScreenCapture
        return MockScreenCapture(target_title=target_title)


def get_input_driver(dry_run: bool = False) -> BaseInputDriver:
    """Dynamically returns the appropriate input driver for the current OS."""
    override = os.environ.get("FLYBRAIN_PLATFORM", "").lower()

    if override == "mock":
        from .mock import MockInputDriver
        return MockInputDriver(dry_run=dry_run)

    if sys.platform == "win32":
        from .windows import WindowsInputDriver
        return WindowsInputDriver(dry_run=dry_run)
    elif sys.platform.startswith("linux"):
        from .linux import LinuxInputDriver
        return LinuxInputDriver(dry_run=dry_run)
    else:
        logger.warning(f"Unsupported OS '{sys.platform}'. Defaulting to MockInputDriver.")
        from .mock import MockInputDriver
        return MockInputDriver(dry_run=dry_run)


def get_platform_drivers(target_title: str = "Half-Life", dry_run: bool = False) -> Tuple[BaseScreenCapture, BaseInputDriver]:
    """Returns both capture and input drivers in a single call."""
    return get_screen_capture_driver(target_title=target_title), get_input_driver(dry_run=dry_run)
