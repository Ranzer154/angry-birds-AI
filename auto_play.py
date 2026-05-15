# auto_play.py
# ─────────────────────────────────────────────────────────────
# FULLY AUTOMATIC ANGRY BIRDS BOT
# 
# HOW IT WORKS:
# 1. Gives you time to unpause the game
# 2. Captures screen and detects everything
# 3. Waits until detection is stable
# 4. Automatically fires at best target
# 5. Waits for outcome
# 6. Repeats until no targets remain
#
# YOU DON'T NEED TO CLICK ANYTHING AFTER STARTING!
#
# EMERGENCY STOP: move mouse to TOP-LEFT corner of screen
# ─────────────────────────────────────────────────────────────

import time
import mss
import numpy as np
import cv2
import pyautogui

from src.vision.detector       import PigDetector
from src.vision.sling_detector import SlingDetector
from src.aiming.trajectory     import TrajectoryCalculator
from src.automation.shooter    import Shooter
from src.utils.config          import CAPTURE_REGION
from src.utils.logger          import get_logger

# Set logger to WARNING to keep terminal clean
import logging
logging.getLogger().setLevel(logging.WARNING)

logger = get_logger("auto_play")

# ── SETTINGS ──────────────────────────────────────────────────
STARTUP_DELAY        = 6    # seconds to unpause game after running
DETECTION_STABLE_TIME = 2.0  # seconds detection must be stable before shooting
MIN_CONFIDENCE       = 0.1  # minimum detection confidence to shoot
BETWEEN_SHOTS_DELAY  = 4.0  # seconds to wait between shots
MAX_SHOTS            = 20   # safety limit — stop after this many shots
DETECTION_SAMPLES    = 5    # how many frames to sample for stable detection


def countdown(seconds: int, message: str) -> None:
    """Shows a live countdown in the terminal."""
    for i in range(seconds, 0, -1):
        print(f"\r{message} {i}s...  ", end="", flush=True)
        time.sleep(1)
    print(f"\r{message} GO!        ")


def capture_frame(sct) -> np.ndarray:
    """Captures one frame from the game."""
    raw   = sct.grab(CAPTURE_REGION)
    frame = np.array(raw)[:, :, :3]
    return frame


def get_stable_detection(
    sct,
    pig_det:   PigDetector,
    sling_det: SlingDetector,
    samples:   int = DETECTION_SAMPLES
) -> tuple:
    """
    Takes multiple detection samples and returns
    the most consistent result.

    WHY? A single frame might miss a pig or have a
    false positive. Averaging across frames gives
    more reliable results.

    Returns:
        (targets, sling_pos) or ([], None) if unstable
    """
    print("  Analyzing scene...", end="", flush=True)

    all_target_counts = []
    all_sling_positions = []
    last_targets  = []
    last_sling    = None

    for i in range(samples):
        frame    = capture_frame(sct)
        targets  = pig_det.detect_all_targets(frame)
        sling    = sling_det.detect(frame)

        all_target_counts.append(len(targets))
        if sling:
            all_sling_positions.append(sling)

        last_targets = targets
        last_sling   = sling

        time.sleep(0.15)
        print(".", end="", flush=True)

    print()

    # Check consistency — target count should be stable
    if not all_target_counts:
        return [], None

    most_common_count = max(
        set(all_target_counts),
        key=all_target_counts.count
    )

    # If detections are wildly inconsistent, wait more
    count_variance = max(all_target_counts) - min(all_target_counts)
    if count_variance > 3:
        print(f"  Detection unstable (variance={count_variance}). Waiting...")
        return [], None

    # Average sling position for stability
    if all_sling_positions:
        avg_sling_x = int(np.mean([p[0] for p in all_sling_positions]))
        avg_sling_y = int(np.mean([p[1] for p in all_sling_positions]))
        stable_sling = (avg_sling_x, avg_sling_y)
    else:
        stable_sling = None

    print(f"  Detected: {most_common_count} targets | "
          f"Sling: {stable_sling}")

    return last_targets, stable_sling


def choose_best_target(targets: list) -> object:
    """
    Chooses the best target to shoot at.

    Priority order:
    1. TNT boxes (cause chain explosions)
    2. Pigs with highest confidence
    3. Leftmost target (closest to slingshot)
    """
    if not targets:
        return None

    # Separate by type
    tnts = [t for t in targets if t.label == "tnt"]
    pigs = [t for t in targets if t.label == "pig"]

    # Priority 1: TNT boxes
    if tnts:
        # Pick TNT closest to center of pig cluster
        if pigs:
            avg_pig_x = np.mean([p.center_x for p in pigs])
            avg_pig_y = np.mean([p.center_y for p in pigs])
            # TNT closest to pig cluster center
            best_tnt = min(
                tnts,
                key=lambda t: abs(t.center_x - avg_pig_x) +
                              abs(t.center_y - avg_pig_y)
            )
            return best_tnt
        return tnts[0]

    # Priority 2: pig with highest confidence
    if pigs:
        return max(pigs, key=lambda p: p.confidence)

    return targets[0]


def print_status(
    shot_num:  int,
    target:    object,
    aim_info:  dict,
    sling_pos: tuple
) -> None:
    """Prints a clean status summary before each shot."""
    print("\n" + "="*50)
    print(f"  SHOT #{shot_num}")
    print("="*50)
    print(f"  Target:   {target.label} at {target.center}")
    print(f"  Sling:    {sling_pos}")
    print(f"  Angle:    {aim_info['angle']}°")
    print(f"  Distance: {aim_info['distance']}px")
    print(f"  Advice:   {aim_info['advice']}")
    print("="*50)


def run_auto_bot():
    """
    Main auto-play loop.
    Runs completely hands-free after startup.
    """

    print("\n" + "█"*50)
    print("  ANGRY BIRDS AI — FULLY AUTO MODE")
    print("█"*50)
    print("\n  EMERGENCY STOP: move mouse to TOP-LEFT corner!")
    print("  Starting up...\n")

    # Give user time to switch to game and unpause
    countdown(STARTUP_DELAY, "Unpause Angry Birds in")

    # Initialize components
    sct       = mss.MSS()
    pig_det   = PigDetector()
    sling_det = SlingDetector()
    traj      = TrajectoryCalculator()
    shooter   = Shooter(capture_region=CAPTURE_REGION)

    shots_fired   = 0
    failed_attempts = 0
    max_failures  = 5  # stop if detection fails too many times

    print("\nBot is now running fully automatically!")
    print("Do NOT touch the mouse — it will be controlled by the bot!\n")

    # ── MAIN AUTO LOOP ────────────────────────────────────────
    while shots_fired < MAX_SHOTS:

        print(f"\n[Round {shots_fired + 1}] Scanning for targets...")

        # ── STABLE DETECTION ──────────────────────────────────
        targets, sling_pos = get_stable_detection(
            sct, pig_det, sling_det
        )

        # ── CHECK IF LEVEL IS CLEAR ───────────────────────────
        if not targets:
            failed_attempts += 1
            print(f"  No targets found! "
                  f"(attempt {failed_attempts}/{max_failures})")

            if failed_attempts >= max_failures:
                print("\n  Level appears CLEAR or bot is stuck!")
                print("  Stopping auto mode.")
                break

            print(f"  Waiting 2s and trying again...")
            time.sleep(2)
            continue

        failed_attempts = 0  # reset on successful detection

        # ── CHECK SLINGSHOT ───────────────────────────────────
        if not sling_pos:
            print("  Slingshot not found! Waiting...")
            time.sleep(2)
            continue

        # ── CHOOSE BEST TARGET ────────────────────────────────
        target = choose_best_target(targets)

        # ── CALCULATE AIM ─────────────────────────────────────
        aim_info = traj.get_aim_info(
            sling  = sling_pos,
            target = target.center
        )

        if not aim_info["reachable"]:
            print(f"  Target unreachable! Trying next target...")
            # Try other targets
            for t in targets:
                if t == target:
                    continue
                aim_info = traj.get_aim_info(sling_pos, t.center)
                if aim_info["reachable"]:
                    target = t
                    break
            else:
                print("  No reachable targets! Waiting...")
                time.sleep(2)
                continue

        # ── PRINT STATUS ──────────────────────────────────────
        print_status(shots_fired + 1, target, aim_info, sling_pos)

        # ── COUNTDOWN BEFORE SHOT ─────────────────────────────
        print(f"\n  Firing in ", end="", flush=True)
        for i in range(3, 0, -1):
            print(f"{i}...", end="", flush=True)
            time.sleep(1)
        print("FIRE!\n")

        # ── EXECUTE SHOT ──────────────────────────────────────
        success = shooter.shoot(
            sling_pos = sling_pos,
            angle_deg = aim_info["angle"],
            power_pct = 1.0,
            dry_run   = False
        )

        if success:
            shots_fired += 1
            print(f"  Shot #{shots_fired} fired!")

        # ── WAIT FOR OUTCOME ──────────────────────────────────
        print(f"  Waiting {BETWEEN_SHOTS_DELAY}s for outcome...")
        time.sleep(BETWEEN_SHOTS_DELAY)

        # ── CHECK IF LEVEL COMPLETE ───────────────────────────
        print("  Checking if level is complete...")
        frame        = capture_frame(sct)
        check_targets = pig_det.detect_all_targets(frame)
        pig_check    = [t for t in check_targets if t.label == "pig"]

        if not pig_check:
            print("\n" + "🎉"*20)
            print("  ALL PIGS ELIMINATED — LEVEL COMPLETE!")
            print("🎉"*20)
            break

        remaining = len(pig_check)
        print(f"  {remaining} pig(s) remaining. Continuing...")

    # ── SESSION SUMMARY ───────────────────────────────────────
    print("\n" + "="*50)
    print("  AUTO BOT SESSION COMPLETE")
    print("="*50)
    print(f"  Total shots fired: {shots_fired}")
    print(f"  Max shots limit:   {MAX_SHOTS}")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_auto_bot()