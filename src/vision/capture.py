# src/vision/capture.py

import mss
import numpy as np
import cv2
from typing import Optional
from src.utils.config import CAPTURE_REGION
from src.utils.logger import get_logger

logger = get_logger("vision.capture")


class ScreenCapture:
    """
    Captures frames from the screen using mss.
    mss is 10x faster than pyautogui.screenshot()
    """

    def __init__(self, region: Optional[dict] = None):
        self.region = region or CAPTURE_REGION
        self._sct = mss.mss()
        self._frame_count = 0
        logger.info(f"ScreenCapture ready. Region: {self.region}")

    def capture(self) -> np.ndarray:
        """
        Grabs one frame.
        Returns BGR NumPy array of shape (height, width, 3)
        """
        raw   = self._sct.grab(self.region)
        frame = np.array(raw)
        frame = frame[:, :, :3]  # drop alpha channel
        self._frame_count += 1
        return frame

    def update_region(self, region: dict) -> None:
        self.region = region
        logger.info(f"Region updated: {region}")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __del__(self):
        if hasattr(self, '_sct'):
            self._sct.close()