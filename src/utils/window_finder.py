# src/utils/window_finder.py

import ctypes
import ctypes.wintypes
from src.utils.logger import get_logger

logger = get_logger("utils.window_finder")

user32 = ctypes.windll.user32


def find_window_by_title(title_substring: str) -> dict | None:
    """
    Finds a window by searching for title_substring in its title.
    Returns capture region dict or None if not found.
    """
    found_hwnd = None

    def enum_callback(hwnd, _):
        nonlocal found_hwnd

        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        if title_substring.lower() in title.lower():
            logger.info(f"Found window: '{title}'")
            found_hwnd = hwnd
            return False

        return True

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM
    )

    user32.EnumWindows(EnumWindowsProc(enum_callback), 0)

    if found_hwnd is None:
        logger.warning(f"No window found with title: '{title_substring}'")
        return None

    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(found_hwnd, ctypes.byref(rect))

    region = {
        "top":    rect.top,
        "left":   rect.left,
        "width":  rect.right - rect.left,
        "height": rect.bottom - rect.top
    }

    logger.info(f"Window region: {region}")
    return region


def get_angry_birds_region() -> dict:
    """
    Tries to find Angry Birds window automatically.
    Falls back to config defaults if not found.
    """
    from src.utils.config import CAPTURE_REGION

    titles_to_try = ["Angry Birds", "angry birds", "AngryBirds"]

    for title in titles_to_try:
        region = find_window_by_title(title)
        if region:
            return region

    logger.warning("Could not find game window. Using config defaults.")
    return CAPTURE_REGION