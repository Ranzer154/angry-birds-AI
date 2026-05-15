# main.py — Milestone 3
# Now includes mouse automation and actual shooting!
#
# KEYBOARD CONTROLS:
#   Q = quit
#   S = save screenshot
#   D = toggle debug mask
#   T = toggle trajectory
#   N = next target
#   F = fire shot (manual trigger)
#   A = toggle AUTO mode (bot shoots automatically)
#   P = preview shot (dry run, no mouse movement)

import cv2
import os
import time
from datetime import datetime

from src.vision.capture        import ScreenCapture
from src.vision.detector       import PigDetector
from src.vision.sling_detector import SlingDetector
from src.aiming.trajectory     import TrajectoryCalculator
from src.aiming.visualizer     import AimVisualizer
from src.automation.shooter    import Shooter
from src.utils.window_finder   import get_angry_birds_region
from src.utils.config          import (
    DISPLAY_SCALE, WINDOW_NAME,
    SCREENSHOTS_DIR, CAPTURE_REGION
)
from src.utils.logger import get_logger

logger = get_logger("main")


def run():

    logger.info("="*50)
    logger.info("Angry Birds AI — Milestone 3: SHOOTING")
    logger.info("="*50)

    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    # ── INITIALIZE ALL COMPONENTS ─────────────────────────────
    region     = get_angry_birds_region()
    capture    = ScreenCapture(region=region)
    pig_det    = PigDetector()
    sling_det  = SlingDetector()
    trajectory = TrajectoryCalculator()
    visualizer = AimVisualizer()
    shooter    = Shooter(capture_region=CAPTURE_REGION)

    # ── STATE ─────────────────────────────────────────────────
    show_mask    = False
    show_traj    = True
    auto_mode    = False   # when True, bot shoots automatically
    frame_count  = 0
    start_time   = time.time()
    target_index = 0
    last_shot_time = 0
    auto_shot_interval = 5.0  # seconds between auto shots
    print("\nStarting in 3 seconds...")
    print("Click on the Angry Birds window NOW to unpause!")
    time.sleep(3)
    logger.info(
        "Keys: Q=quit F=fire A=auto-mode "
        "N=next-target P=preview T=traj D=mask S=save"
    )
    print("\n" + "="*50)
    print("IMPORTANT SAFETY INFO")
    print("="*50)
    print("Move mouse to TOP-LEFT corner to emergency stop!")
    print("Press A to toggle auto-shooting mode")
    print("Press F to fire ONE shot manually")
    print("Press P to preview shot without moving mouse")
    print("="*50 + "\n")

    while True:

        # ── 1. CAPTURE ────────────────────────────────────────
        frame = capture.capture()
        frame_count += 1

        # ── 2. DETECT EVERYTHING ──────────────────────────────
        all_targets = pig_det.detect_all_targets(frame)
        pigs        = [t for t in all_targets if t.label == "pig"]
        tnts        = [t for t in all_targets if t.label == "tnt"]
        sling_pos   = sling_det.detect(frame)

        # ── 3. CALCULATE AIM ──────────────────────────────────
        aim_info = None
        target   = None

        if sling_pos and all_targets:
            target_index = target_index % len(all_targets)
            target       = all_targets[target_index]

            aim_info = trajectory.get_aim_info(
                sling  = sling_pos,
                target = target.center
            )

        # ── 4. AUTO SHOOTING ──────────────────────────────────
        current_time = time.time()
        time_since_shot = current_time - last_shot_time

        if (auto_mode
                and aim_info
                and aim_info["reachable"]
                and sling_pos
                and target
                and time_since_shot > auto_shot_interval):

            logger.info("AUTO MODE: Firing shot!")
            shooter.shoot(
                sling_pos = sling_pos,
                angle_deg = aim_info["angle"],
                power_pct = 1.0
            )
            last_shot_time = time.time()

        # ── 5. VISUALIZE ──────────────────────────────────────
        if show_mask:
            display = pig_det.get_mask_visual(frame)
        else:
            display = frame.copy()
            display = pig_det.visualize(display, all_targets)
            display = sling_det.visualize(display, sling_pos)

            if show_traj and aim_info and aim_info["reachable"]:
                display = visualizer.draw_trajectory(
                    display, aim_info["path"]
                )
                display = visualizer.draw_aim_line(
                    display, sling_pos, target.center
                )
                display = visualizer.draw_angle_info(
                    display, sling_pos,
                    aim_info["angle"],
                    aim_info["distance"]
                )
                if target:
                    display = visualizer.draw_target_marker(
                        display, target.center
                    )

        # ── 6. OVERLAYS ───────────────────────────────────────
        elapsed = time.time() - start_time
        fps     = frame_count / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            display, f"FPS: {fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 200, 255), 2
        )

        # Auto mode indicator
        auto_text  = "AUTO: ON" if auto_mode else "AUTO: OFF"
        auto_color = (0, 255, 0) if auto_mode else (0, 0, 255)
        cv2.putText(
            display, auto_text,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, auto_color, 2
        )

        # Target info
        if target:
            cv2.putText(
                display,
                f"Target: {target.label} at {target.center}",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )

        # Shot counter
        cv2.putText(
            display,
            f"Shots: {shooter.shots_fired}",
            (10, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        # Next shot countdown in auto mode
        if auto_mode:
            countdown = max(
                0,
                auto_shot_interval - time_since_shot
            )
            cv2.putText(
                display,
                f"Next shot in: {countdown:.1f}s",
                (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )

        # ── 7. SCALE AND SHOW ─────────────────────────────────
        h, w    = display.shape[:2]
        display = cv2.resize(
            display,
            (int(w * DISPLAY_SCALE),
             int(h * DISPLAY_SCALE))
        )
        cv2.imshow(WINDOW_NAME, display)

        # ── 8. KEYBOARD ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            logger.info("Quit.")
            break

        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOTS_DIR, f"frame_{ts}.png")
            cv2.imwrite(path, frame)
            logger.info(f"Saved: {path}")

        elif key == ord('d'):
            show_mask = not show_mask

        elif key == ord('t'):
            show_traj = not show_traj

        elif key == ord('n'):
            if all_targets:
                target_index = (target_index + 1) % len(all_targets)
                logger.info(f"Target: {target_index + 1}")

        elif key == ord('p'):
            # Preview shot — no mouse movement
            if sling_pos and aim_info and aim_info["reachable"]:
                shooter.test_shot(sling_pos, aim_info["angle"])
            else:
                print("Cannot preview — no valid aim calculated")

        elif key == ord('f'):
            # Fire ONE manual shot
            if sling_pos and aim_info and aim_info["reachable"]:
                logger.info("MANUAL SHOT triggered!")
                shooter.shoot(
                    sling_pos = sling_pos,
                    angle_deg = aim_info["angle"],
                    power_pct = 1.0
                )
                last_shot_time = time.time()
            else:
                print("Cannot shoot — no valid aim calculated")

        elif key == ord('a'):
            # Toggle auto mode
            auto_mode = not auto_mode
            status    = "ENABLED" if auto_mode else "DISABLED"
            logger.info(f"Auto mode: {status}")
            print(f"\nAuto shooting: {status}")
            if auto_mode:
                print("Bot will shoot automatically every "
                      f"{auto_shot_interval}s")
                print("Move mouse to top-left corner to stop!")

    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    logger.info(
        f"Session ended. "
        f"{frame_count} frames | "
        f"{shooter.shots_fired} shots | "
        f"{total_time:.1f}s"
    )


if __name__ == "__main__":
    run()