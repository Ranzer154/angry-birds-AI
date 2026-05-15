# src/automation/shooter.py
# ─────────────────────────────────────────────────────────────
# High-level shooting logic.
#
# This class takes:
# - slingshot position (from SlingDetector)
# - target position (from PigDetector)
# - angle (from TrajectoryCalculator)
#
# And converts them into actual mouse drag actions.
#
# COORDINATE CONVERSION:
# Our detection uses GAME coordinates (relative to capture region)
# Mouse automation needs SCREEN coordinates (absolute pixels)
# We must add the capture region offset to convert between them.
# ─────────────────────────────────────────────────────────────

import time
import numpy as np
from typing import Tuple, Optional
from src.automation.mouse_control import MouseController
from src.utils.config import CAPTURE_REGION
from src.utils.logger import get_logger

logger = get_logger("automation.shooter")

# ── SLINGSHOT SETTINGS ────────────────────────────────────────
# Maximum pixels we can pull back the slingshot
# Pulling too far has no extra effect — game caps it
MAX_PULL_DISTANCE = 85

# Minimum pull distance for a valid shot
MIN_PULL_DISTANCE = 30

# How long to wait after shot before detecting again (seconds)
POST_SHOT_DELAY = 3.0

# How long to wait for bird to settle in slingshot (seconds)
PRE_SHOT_DELAY = 1.0


class Shooter:
    """
    Converts aim calculations into actual mouse actions.

    The shooting sequence:
    1. Wait for bird to settle in slingshot
    2. Calculate pull-back position
    3. Move mouse to bird position
    4. Drag backward (pull back)
    5. Release (shoot)
    6. Wait for outcome
    """

    def __init__(
        self,
        capture_region: dict = None,
        move_duration:  float = 0.3,
        drag_duration:  float = 0.5
    ):
        self.region = capture_region or CAPTURE_REGION
        self.mouse  = MouseController(
            move_duration=move_duration,
            drag_duration=drag_duration
        )
        self.shots_fired = 0
        logger.info("Shooter initialized")

    # ── COORDINATE CONVERSION ─────────────────────────────────

    def game_to_screen(
        self,
        game_x: int,
        game_y: int
    ) -> Tuple[int, int]:
        """
        Converts game coordinates to screen coordinates.

        WHY NEEDED?
        Our CV detects objects relative to the capture region.
        Example: pig at game position (500, 300)
        If capture region starts at screen (15, 45):
        Screen position = (15 + 500, 45 + 300) = (515, 345)

        Args:
            game_x, game_y: position within the captured frame

        Returns:
            (screen_x, screen_y): absolute screen position
        """
        screen_x = self.region["left"] + game_x
        screen_y = self.region["top"]  + game_y
        return (screen_x, screen_y)

    # ── PULL-BACK CALCULATION ─────────────────────────────────

    def calculate_pullback(
        self,
        sling_pos:  Tuple[int, int],
        angle_deg:  float,
        power_pct:  float = 1.0
    ) -> Tuple[int, int]:
        """
        Calculates where to drag the mouse to achieve
        the desired launch angle.

        The pull-back direction is OPPOSITE to the launch direction.

        Args:
            sling_pos:  (x, y) slingshot tip in game coordinates
            angle_deg:  launch angle in degrees (0° = horizontal right)
            power_pct:  power percentage 0.0 to 1.0

        Returns:
            (x, y) pull-back position in game coordinates
        """
        sx, sy    = sling_pos
        angle_rad = np.radians(angle_deg)

        # Pull distance scales with power percentage
        pull_dist = MAX_PULL_DISTANCE * max(
            0.0, min(1.0, power_pct)
        )
        pull_dist = max(pull_dist, MIN_PULL_DISTANCE)

        # Pull direction is OPPOSITE to launch direction
        # Launch: right and up → Pull: left and down
        pull_x = sx - int(pull_dist * np.cos(angle_rad))
        pull_y = sy + int(pull_dist * np.sin(angle_rad))
        # ↑ Note: + for y because screen y increases downward

        logger.debug(
            f"Pull-back: angle={angle_deg:.1f}° "
            f"dist={pull_dist:.0f}px "
            f"position=({pull_x},{pull_y})"
        )
        return (pull_x, pull_y)

    # ── SHOOTING ──────────────────────────────────────────────

    def shoot(
        self,
        sling_pos:  Tuple[int, int],
        angle_deg:  float,
        power_pct:  float = 1.0,
        dry_run:    bool  = False
    ) -> bool:
        """
        Executes a complete shot.

        Args:
            sling_pos: slingshot position in game coordinates
            angle_deg: calculated launch angle
            power_pct: shot power 0.0 to 1.0
            dry_run:   if True, log but don't actually move mouse
                       Use this to test without affecting game!

        Returns:
            True if shot was executed, False if skipped
        """
        self.shots_fired += 1
        logger.info(
            f"Shot #{self.shots_fired} | "
            f"angle={angle_deg:.1f}° power={power_pct:.0%}"
        )

        # Calculate pull-back position in game coords
        pullback_game = self.calculate_pullback(
            sling_pos, angle_deg, power_pct
        )

        # Convert both positions to screen coordinates
        sling_screen   = self.game_to_screen(*sling_pos)
        pullback_screen = self.game_to_screen(*pullback_game)

        logger.info(f"Sling screen pos:   {sling_screen}")
        logger.info(f"Pullback screen pos: {pullback_screen}")

        if dry_run:
            logger.info("DRY RUN — no mouse movement")
            return False

        # Wait for bird to settle
        logger.info(f"Waiting {PRE_SHOT_DELAY}s before shooting...")
        time.sleep(PRE_SHOT_DELAY)

        # Execute the drag
        self.mouse.drag_to(
            start_x = sling_screen[0],
            start_y = sling_screen[1],
            end_x   = pullback_screen[0],
            end_y   = pullback_screen[1]
        )

        # Wait for bird to land and outcome to resolve
        logger.info(
            f"Shot fired! Waiting {POST_SHOT_DELAY}s "
            f"for outcome..."
        )
        time.sleep(POST_SHOT_DELAY)

        return True

    def test_shot(
        self,
        sling_pos: Tuple[int, int],
        angle_deg: float
    ) -> None:
        """
        Logs what a shot WOULD do without moving the mouse.
        Use this first to verify calculations before real shots.
        """
        pullback = self.calculate_pullback(sling_pos, angle_deg)
        sling_screen   = self.game_to_screen(*sling_pos)
        pullback_screen = self.game_to_screen(*pullback)

        print(f"\n{'─'*40}")
        print(f"SHOT PREVIEW (dry run)")
        print(f"{'─'*40}")
        print(f"Angle:           {angle_deg:.1f}°")
        print(f"Sling (game):    {sling_pos}")
        print(f"Pullback (game): {pullback}")
        print(f"Sling (screen):  {sling_screen}")
        print(f"Pullback (screen): {pullback_screen}")
        print(f"{'─'*40}\n")