# src/aiming/visualizer.py
# ─────────────────────────────────────────────────────────────
# Draws trajectory lines and aiming information on screen.
#
# WHY SEPARATE FROM TRAJECTORY CALCULATOR?
# Single Responsibility Principle — one class does one job.
# Calculator does math. Visualizer draws things.
# This makes both easier to test and modify.
# ─────────────────────────────────────────────────────────────

import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger("aiming.visualizer")


class AimVisualizer:
    """
    Draws trajectory paths, angles, and aiming info on frames.
    """

    def draw_trajectory(
        self,
        frame:  np.ndarray,
        points: List[Tuple[int, int]],
        color:  Tuple[int, int, int] = (255, 255, 0),
        dotted: bool = True
    ) -> np.ndarray:
        """
        Draws the trajectory path as a dotted or solid line.

        Args:
            frame:  BGR frame to draw on (we copy it first)
            points: list of (x,y) points along the path
            color:  BGR color for the line
            dotted: if True, draw dots instead of solid line

        Returns:
            annotated frame copy
        """
        output = frame.copy()

        if len(points) < 2:
            return output

        if dotted:
            # Draw every 3rd point as a small circle
            # Creates a dashed/dotted effect
            for i, (x, y) in enumerate(points):
                if i % 3 == 0:
                    cv2.circle(output, (x, y), 3, color, -1)
        else:
            # Draw solid line connecting all points
            for i in range(len(points) - 1):
                cv2.line(output, points[i], points[i+1], color, 2)

        # Draw a larger dot at the start (launch point)
        if points:
            cv2.circle(output, points[0], 6, (0, 255, 255), -1)

        return output

    def draw_aim_line(
        self,
        frame:  np.ndarray,
        sling:  Tuple[int, int],
        target: Tuple[int, int]
    ) -> np.ndarray:
        """
        Draws a straight line from sling to target.
        This shows the DIRECT line, not the actual curved path.
        """
        output = frame.copy()
        cv2.line(output, sling, target, (255, 0, 255), 1)
        return output

    def draw_angle_info(
        self,
        frame:     np.ndarray,
        sling:     Tuple[int, int],
        angle_deg: Optional[float],
        distance:  float
    ) -> np.ndarray:
        """
        Draws angle and distance information on the frame.
        """
        output = frame.copy()

        if angle_deg is None:
            text = "UNREACHABLE"
            color = (0, 0, 255)
        else:
            text  = f"Angle: {angle_deg:.1f} deg  Dist: {distance:.0f}px"
            color = (255, 255, 0)

        cv2.putText(
            output, text,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        # Draw angle arc at slingshot position
        if angle_deg is not None and sling:
            sx, sy    = sling
            angle_rad = np.radians(angle_deg)

            # Draw a short line showing the aim direction
            length = 60
            end_x  = int(sx + length * np.cos(angle_rad))
            end_y  = int(sy - length * np.sin(angle_rad))
            cv2.arrowedLine(
                output,
                (sx, sy),
                (end_x, end_y),
                (0, 255, 255),
                2,
                tipLength=0.3
            )

        return output

    def draw_target_marker(
        self,
        frame:  np.ndarray,
        target: Tuple[int, int]
    ) -> np.ndarray:
        """
        Draws a crosshair marker on the target pig.
        """
        output = frame.copy()
        x, y   = target

        # Outer circle
        cv2.circle(output, (x, y), 20, (0, 0, 255), 2)
        # Inner dot
        cv2.circle(output, (x, y), 4,  (0, 0, 255), -1)
        # Crosshair lines
        cv2.line(output, (x-25, y), (x+25, y), (0, 0, 255), 1)
        cv2.line(output, (x, y-25), (x, y+25), (0, 0, 255), 1)

        return output