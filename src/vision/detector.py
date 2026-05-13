# src/vision/detector.py
# Updated with shape filtering + TNT detection

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from src.utils.config import PIG_HSV_LOWER, PIG_HSV_UPPER, MIN_PIG_AREA
from src.utils.logger import get_logger

logger = get_logger("vision.detector")

# ── TNT COLOR RANGE ───────────────────────────────────────────
# Tuned from real game samples
TNT_HSV_LOWER = (13, 140, 170)
TNT_HSV_UPPER = (20, 255, 255)
MIN_TNT_AREA  = 400


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

    @property
    def is_high_value(self) -> bool:
        return self.label == "tnt"

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
        return (
            f"[{self.label}] center=({cx},{cy}) "
            f"size={self.width}x{self.height} "
            f"conf={self.confidence:.2f}"
        )


class PigDetector:
    """
    Detects pigs and TNT boxes using HSV color
    segmentation + shape filtering.

    Shape filtering explanation:
    - Pigs are circular  → we measure circularity
    - TNT is rectangular → we measure rectangularity
    - This removes false positives like grass and sky
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

        self.tnt_lower = np.array(TNT_HSV_LOWER, dtype=np.uint8)
        self.tnt_upper = np.array(TNT_HSV_UPPER, dtype=np.uint8)

        logger.info(
            f"PigDetector ready | "
            f"HSV: {hsv_lower}→{hsv_upper} | "
            f"min_area: {min_area}"
        )

    # ── SHAPE HELPERS ─────────────────────────────────────────

    def _get_circularity(self, contour) -> float:
        """
        How circular is this contour?
        Perfect circle = 1.0
        Pigs should score above 0.35
        """
        area      = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter ** 2)

    def _get_rectangularity(self, contour) -> float:
        """
        How rectangular is this contour?
        Perfect rectangle = 1.0
        TNT boxes should score above 0.6
        """
        area            = cv2.contourArea(contour)
        x, y, w, h      = cv2.boundingRect(contour)
        bbox_area       = w * h
        if bbox_area == 0:
            return 0.0
        return area / bbox_area

    # ── CORE DETECTION ────────────────────────────────────────

    def _detect_objects(
        self,
        frame:        np.ndarray,
        hsv_lower:    np.ndarray,
        hsv_upper:    np.ndarray,
        label:        str,
        min_area:     int,
        shape_filter: str   = "circle",
        min_shape:    float = 0.35
    ) -> List[Detection]:
        """
        Generic detection pipeline for any object type.

        Steps:
        1. Convert to HSV
        2. Create color mask
        3. Clean mask with morphology
        4. Find contours
        5. Filter by area
        6. Filter by shape
        7. Return as Detection objects
        """
        # Step 1 and 2
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # Step 3 - clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.erode(mask,  kernel, iterations=1)
        mask   = cv2.dilate(mask, kernel, iterations=2)

        # Step 4
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Step 5 - area filter
            if area < min_area:
                continue

            # Step 6 - shape filter
            if shape_filter == "circle":
                shape_score = self._get_circularity(contour)
            else:
                shape_score = self._get_rectangularity(contour)

            if shape_score < min_shape:
                continue

            # Step 7 - create detection
            x, y, w, h = cv2.boundingRect(contour)
            confidence  = min(1.0, (area / 10000.0) * shape_score)

            detections.append(Detection(
                label=label,
                x=x, y=y,
                width=w, height=h,
                confidence=confidence
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── PUBLIC METHODS ────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detects pigs using color + circularity filter."""
        pigs = self._detect_objects(
            frame        = frame,
            hsv_lower    = self.lower,
            hsv_upper    = self.upper,
            label        = "pig",
            min_area     = self.min_area,
            shape_filter = "circle",
            min_shape    = 0.35
        )
        logger.info(f"Pigs detected: {len(pigs)}")
        return pigs

    def detect_tnt(self, frame: np.ndarray) -> List[Detection]:
        """Detects TNT boxes using color + rectangularity filter."""
        tnts = self._detect_objects(
            frame        = frame,
            hsv_lower    = self.tnt_lower,
            hsv_upper    = self.tnt_upper,
            label        = "tnt",
            min_area     = MIN_TNT_AREA,
            shape_filter = "rectangle",
            min_shape    = 0.6
        )
        logger.info(f"TNT detected: {len(tnts)}")
        return tnts

    def detect_all_targets(
        self,
        frame: np.ndarray
    ) -> List[Detection]:
        """
        Detects ALL targets: TNT boxes + pigs.
        TNT comes first because it has higher strategic value.
        Hitting TNT causes chain explosions that kill nearby pigs.
        """
        pigs = self.detect(frame)
        tnts = self.detect_tnt(frame)

        all_targets = tnts + pigs

        logger.info(
            f"Total targets: {len(all_targets)} "
            f"({len(tnts)} TNT + {len(pigs)} pigs)"
        )
        return all_targets

    # ── VISUALIZATION ─────────────────────────────────────────

    def visualize(
        self,
        frame:        np.ndarray,
        detections:   List[Detection],
        show_centers: bool = True
    ) -> np.ndarray:
        """
        Draws bounding boxes on frame.
        Green boxes = pigs
        Red boxes   = TNT
        """
        output = frame.copy()

        colors = {
            "pig": (0, 255, 0),   # green
            "tnt": (0, 0, 255),   # red
        }

        pig_count = sum(1 for d in detections if d.label == "pig")
        tnt_count = sum(1 for d in detections if d.label == "tnt")

        for i, det in enumerate(detections):
            color = colors.get(det.label, (255, 255, 0))

            # Bounding box
            cv2.rectangle(
                output,
                (det.x, det.y),
                det.bottom_right,
                color, 2
            )

            # Label background
            label = f"{det.label} ({det.confidence:.2f})"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                output,
                (det.x, det.y - th - 8),
                (det.x + tw + 4, det.y),
                color, -1
            )

            # Label text
            cv2.putText(
                output, label,
                (det.x + 2, det.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 1
            )

            # Center dot
            if show_centers:
                cv2.circle(output, det.center, 5, color, -1)

        # Count overlay
        cv2.putText(
            output,
            f"Pigs: {pig_count}  TNT: {tnt_count}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        return output

    def get_mask_visual(self, frame: np.ndarray) -> np.ndarray:
        """Returns the pig color mask for debugging."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)