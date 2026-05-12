# src/vision/detector.py

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from src.utils.config import PIG_HSV_LOWER, PIG_HSV_UPPER, MIN_PIG_AREA
from src.utils.logger import get_logger

logger = get_logger("vision.detector")


@dataclass
class Detection:
    label:      str
    x:          int
    y:          int
    width:      int
    height:     int
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (self.x + self.width, self.y + self.height)

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "x":          self.x,
            "y":          self.y,
            "width":      self.width,
            "height":     self.height,
            "center_x":   self.center_x,
            "center_y":   self.center_y,
            "confidence": round(self.confidence, 3)
        }

    def __str__(self) -> str:
        cx, cy = self.center
        return (f"[{self.label}] center=({cx},{cy}) "
                f"size={self.width}x{self.height} "
                f"conf={self.confidence:.2f}")


class PigDetector:
    """
    Detects pigs using HSV color segmentation.
    Phase 1 detector — upgrades to YOLO in Phase 2.
    """

    def __init__(
        self,
        hsv_lower: tuple = PIG_HSV_LOWER,
        hsv_upper: tuple = PIG_HSV_UPPER,
        min_area:  int   = MIN_PIG_AREA
    ):
        self.lower    = np.array(hsv_lower, dtype=np.uint8)
        self.upper    = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        logger.info(f"PigDetector ready | HSV: {hsv_lower} → {hsv_upper}")

    def _to_hsv(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def _make_mask(self, hsv: np.ndarray) -> np.ndarray:
        mask   = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.erode(mask, kernel, iterations=1)
        mask   = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def _get_contours(self, mask: np.ndarray) -> list:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def detect(self, frame: np.ndarray) -> List[Detection]:
        hsv      = self._to_hsv(frame)
        mask     = self._make_mask(hsv)
        contours = self._get_contours(mask)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            confidence  = min(1.0, area / 8000.0)

            detections.append(Detection(
                label="pig",
                x=x, y=y,
                width=w, height=h,
                confidence=confidence
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info(f"Pigs detected: {len(detections)}")
        return detections

    def visualize(
        self,
        frame:        np.ndarray,
        detections:   List[Detection],
        show_centers: bool = True
    ) -> np.ndarray:

        output = frame.copy()

        for i, det in enumerate(detections):

            # Bounding box
            cv2.rectangle(
                output,
                (det.x, det.y),
                det.bottom_right,
                (0, 255, 0), 2
            )

            # Label background
            label = f"pig {i+1} ({det.confidence:.2f})"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                output,
                (det.x, det.y - th - 8),
                (det.x + tw + 4, det.y),
                (0, 200, 0), -1
            )

            # Label text
            cv2.putText(
                output, label,
                (det.x + 2, det.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1
            )

            # Center dot
            if show_centers:
                cv2.circle(
                    output, det.center,
                    5, (0, 0, 255), -1
                )

        # Pig count overlay
        cv2.putText(
            output, f"Pigs: {len(detections)}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        return output

    def get_mask_visual(self, frame: np.ndarray) -> np.ndarray:
        hsv  = self._to_hsv(frame)
        mask = self._make_mask(hsv)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)