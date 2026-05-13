# src/aiming/trajectory.py
# ─────────────────────────────────────────────────────────────
# Calculates the trajectory (flight path) of a bird.
#
# PHYSICS LESSON:
# A launched bird follows a parabolic path described by:
#
#   x(t) = x₀ + v₀·cos(θ)·t
#   y(t) = y₀ - v₀·sin(θ)·t + ½·g·t²
#
# Note: y increases DOWNWARD on screen, so gravity is POSITIVE.
#
# To hit a target at (tx, ty) from (sx, sy):
# We solve for the angle θ that makes the parabola pass through target.
#
# The solution uses the quadratic formula on the trajectory equations.
# ─────────────────────────────────────────────────────────────

import numpy as np
import cv2
from typing import Optional, List, Tuple
from src.utils.logger import get_logger

logger = get_logger("aiming.trajectory")

# Game physics constants
# These are approximations — we'll tune them with real shots later
GRAVITY    = 9.8    # pixels per frame² (approximate)
LAUNCH_POWER = 80.0 # initial velocity magnitude (approximate)


class TrajectoryCalculator:
    """
    Calculates launch angles and trajectory paths.

    COORDINATE SYSTEM NOTE:
    Screen coordinates have y increasing DOWNWARD.
    Physics has y increasing UPWARD.
    We handle this conversion internally.
    """

    def __init__(
        self,
        gravity:      float = GRAVITY,
        launch_power: float = LAUNCH_POWER
    ):
        self.gravity      = gravity
        self.launch_power = launch_power
        logger.info(
            f"TrajectoryCalculator | "
            f"gravity={gravity} power={launch_power}"
        )

    def calculate_angle(
        self,
        sling:  Tuple[int, int],
        target: Tuple[int, int]
    ) -> Optional[float]:
        """
        Calculates the launch angle needed to hit the target.

        Args:
            sling:  (x, y) slingshot tip position in screen coords
            target: (x, y) pig center position in screen coords

        Returns:
            angle in degrees, or None if target is unreachable
        """
        sx, sy = sling
        tx, ty = target

        # Convert to physics coordinates (flip y axis)
        # In physics: positive y = up
        # On screen:  positive y = down
        # So we negate the y difference
        dx =  (tx - sx)          # horizontal distance (same in both)
        dy = -(ty - sy)          # vertical distance (flipped!)

        logger.debug(f"Sling→Target: dx={dx}, dy={dy}")

        # If target is to the LEFT, we can't shoot that way
        if dx <= 0:
            logger.warning("Target is behind or at slingshot!")
            return None

        v  = self.launch_power
        g  = self.gravity

        # ── ANGLE FORMULA ─────────────────────────────────────
        # From projectile motion equations, solving for θ:
        #
        # discriminant = v⁴ - g(g·dx² + 2·dy·v²)
        #
        # If discriminant < 0, target is out of range.
        # If discriminant >= 0, there are two solutions:
        #   - low angle (flatter, faster path)
        #   - high angle (looping, slower path)
        # We prefer the LOW angle for most shots.

        v2           = v * v
        v4           = v2 * v2
        discriminant = v4 - g * (g * dx**2 + 2 * dy * v2)

        if discriminant < 0:
            logger.warning(
                f"Target unreachable! "
                f"discriminant={discriminant:.1f}. "
                f"Try increasing LAUNCH_POWER in config."
            )
            return None

        sqrt_disc = np.sqrt(discriminant)

        # Two possible angles
        angle1 = np.arctan2(v2 + sqrt_disc, g * dx)  # high arc
        angle2 = np.arctan2(v2 - sqrt_disc, g * dx)  # low arc

        # Convert from radians to degrees
        angle1_deg = np.degrees(angle1)
        angle2_deg = np.degrees(angle2)

        logger.debug(
            f"Angles: high={angle1_deg:.1f}° low={angle2_deg:.1f}°"
        )

        # Prefer the low angle (more direct shot)
        # But if low angle is negative, use high angle
        if angle2_deg > 0:
            chosen = angle2_deg
        else:
            chosen = angle1_deg

        logger.info(f"Chosen angle: {chosen:.1f}°")
        return chosen

    def simulate_path(
        self,
        sling:      Tuple[int, int],
        angle_deg:  float,
        num_points: int = 60
    ) -> List[Tuple[int, int]]:
        """
        Simulates the trajectory path and returns a list of points.

        These points can be drawn on screen to visualize the shot.

        Args:
            sling:      (x, y) launch position in screen coords
            angle_deg:  launch angle in degrees
            num_points: how many points to calculate along the path

        Returns:
            List of (x, y) screen coordinate points along the path
        """
        sx, sy    = sling
        angle_rad = np.radians(angle_deg)
        v         = self.launch_power
        g         = self.gravity

        # Velocity components
        vx =  v * np.cos(angle_rad)  # horizontal (always positive)
        vy =  v * np.sin(angle_rad)  # vertical (positive = up in physics)

        points = []

        for t in np.linspace(0, num_points * 0.3, num_points):
            # Physics coordinates
            px_phys = sx + vx * t
            py_phys = sy - vy * t + 0.5 * g * t**2
            # ↑ minus vy because screen y increases downward
            # ↑ plus gravity because it pulls down (increases y)

            px = int(px_phys)
            py = int(py_phys)

            # Stop if path goes off screen
            if px < 0 or py < 0:
                break

            points.append((px, py))

        return points

    def get_aim_info(
        self,
        sling:  Tuple[int, int],
        target: Tuple[int, int]
    ) -> dict:
        """
        Returns complete aiming information for a shot.

        Args:
            sling:  slingshot position
            target: pig position

        Returns:
            dict with angle, distance, path points, and recommendation
        """
        sx, sy = sling
        tx, ty = target

        dx       = tx - sx
        dy       = ty - sy
        distance = np.sqrt(dx**2 + dy**2)
        angle    = self.calculate_angle(sling, target)

        if angle is None:
            return {
                "reachable": False,
                "angle":     None,
                "distance":  distance,
                "path":      [],
                "advice":    "Target unreachable — increase LAUNCH_POWER"
            }

        path = self.simulate_path(sling, angle)

        return {
            "reachable": True,
            "angle":     round(angle, 1),
            "distance":  round(distance, 1),
            "path":      path,
            "advice":    f"Aim at {angle:.1f}° | Distance: {distance:.0f}px"
        }