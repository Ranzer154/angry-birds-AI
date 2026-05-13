# src/vision/sling_detector.py
# ─────────────────────────────────────────────────────────────
# Detects the slingshot position in Angry Birds.
#
# WHY DETECT THE SLINGSHOT?
# It is the LAUNCH POINT for every bird.
# All trajectory calculations start from the slingshot tip.
#
# HOW IT WORKS:
# The slingshot is a dark brown/wooden color.
# We use HSV color detection to find it, same as pig detection.
# Then we find the TOP of the slingshot — that's the launch point.
# ─────────────────────────────────────────────────────────────

import cv2
import numpy as np
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("vision.sling_detector")

# ── SLINGSHOT COLOR RANGE (HSV) ───────────────────────────────
# The slingshot is dark brown/wooden
# Hue 10-25 covers orange-brown range
# High saturation, medium-low value (it's quite dark)
SLING_HSV_LOWER = (5,  50,  30)
SLING_HSV_UPPER = (25, 255, 180)

# Minimum area to count as slingshot (pixels)
MIN_SLING_AREA = 200


class SlingDetector:
    """
    Detects the slingshot and returns its tip position.

    The "tip" is the top-center of the slingshot fork —
    this is where the bird sits and where shots launch from.
    """

    def __init__(
        self,
        hsv_lower: tuple = SLING_HSV_LOWER,
        hsv_upper: tuple = SLING_HSV_UPPER,
        min_area:  int   = MIN_SLING_AREA
    ):
        self.lower    = np.array(hsv_lower, dtype=np.uint8)
        self.upper    = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area

        # Cache the last known position.
        # If detection fails for one frame, we use the cached value.
        # The slingshot doesn't move, so this is safe.
        self._cached_position: Optional[Tuple[int, int]] = None

        logger.info("SlingDetector initialized")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Finds the slingshot tip position.

        Args:
            frame: BGR NumPy array from ScreenCapture

        Returns:
            (x, y) pixel position of slingshot tip
            or None if not found
        """
        # Convert to HSV
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create brown mask
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask   = cv2.erode(mask,  kernel, iterations=1)
        mask   = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.debug("No slingshot contours found")
            return self._cached_position

        # The slingshot is on the LEFT side of the screen.
        # Filter contours to only look at left 40% of frame.
        frame_width = frame.shape[1]
        left_zone   = frame_width * 0.25

        sling_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Only consider contours in the left portion of screen
            if x < left_zone:
                sling_contours.append(c)

        if not sling_contours:
            logger.debug("No slingshot in left zone")
            return self._cached_position

        # Take the largest brown contour in the left zone
        largest = max(sling_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # The TIP of the slingshot is at the TOP CENTER of the bounding box
        # This is where the bird actually sits
        tip_x = x + w // 2
        tip_y = y  # top of the bounding box

        self._cached_position = (tip_x, tip_y)
        logger.debug(f"Slingshot tip: ({tip_x}, {tip_y})")
        return (tip_x, tip_y)

    def visualize(
        self,
        frame:    np.ndarray,
        position: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Draws the slingshot position on a copy of the frame.
        """
        output = frame.copy()

        if position is None:
            cv2.putText(
                output, "Sling: NOT FOUND",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )
            return output

        x, y = position

        # Draw a crosshair at the slingshot tip
        size = 15
        cv2.line(output, (x - size, y), (x + size, y), (0, 165, 255), 2)
        cv2.line(output, (x, y - size), (x, y + size), (0, 165, 255), 2)
        cv2.circle(output, (x, y), 8, (0, 165, 255), -1)

        # Label
        cv2.putText(
            output, f"Sling ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 165, 255), 2
        )

        return output