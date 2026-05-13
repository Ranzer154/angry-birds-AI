# main.py — Milestone 2 Complete
# Detects pigs + TNT boxes
# Calculates trajectory to highest value target
# Controls: Q=quit S=save D=mask T=trajectory N=next-target

import cv2
import os
import time
from datetime import datetime

from src.vision.capture        import ScreenCapture
from src.vision.detector       import PigDetector
from src.vision.sling_detector import SlingDetector
from src.aiming.trajectory     import TrajectoryCalculator
from src.aiming.visualizer     import AimVisualizer
from src.utils.window_finder   import get_angry_birds_region
from src.utils.config          import (
    DISPLAY_SCALE, WINDOW_NAME, SCREENSHOTS_DIR
)
from src.utils.logger import get_logger

logger = get_logger("main")


def run():

    logger.info("="*50)
    logger.info("Angry Birds AI — Milestone 2")
    logger.info("="*50)

    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    # ── INITIALIZE ALL COMPONENTS ─────────────────────────────
    region     = get_angry_birds_region()
    capture    = ScreenCapture(region=region)
    pig_det    = PigDetector()
    sling_det  = SlingDetector()
    trajectory = TrajectoryCalculator()
    visualizer = AimVisualizer()

    # ── STATE ─────────────────────────────────────────────────
    show_mask    = False
    show_traj    = True
    frame_count  = 0
    start_time   = time.time()
    target_index = 0

    logger.info("Keys: Q=quit S=save D=mask T=trajectory N=next-target")

    while True:

        # ── 1. CAPTURE ────────────────────────────────────────
        frame = capture.capture()
        frame_count += 1

        # ── 2. DETECT PIGS + TNT ──────────────────────────────
        all_targets = pig_det.detect_all_targets(frame)
        pigs = [t for t in all_targets if t.label == "pig"]
        tnts = [t for t in all_targets if t.label == "tnt"]

        # ── 3. DETECT SLINGSHOT ───────────────────────────────
        sling_pos = sling_det.detect(frame)

        # ── 4. CALCULATE TRAJECTORY ───────────────────────────
        aim_info = None
        target   = None

        if sling_pos and all_targets:
            # Keep target index in bounds
            target_index = target_index % len(all_targets)
            target       = all_targets[target_index]

            aim_info = trajectory.get_aim_info(
                sling  = sling_pos,
                target = target.center
            )

            # Print info every 60 frames
            if frame_count % 60 == 0:
                print(f"\n{'─'*40}")
                print(f"Frame {frame_count}")
                print(f"Sling position : {sling_pos}")
                print(f"Pigs found     : {len(pigs)}")
                print(f"TNT found      : {len(tnts)}")
                print(f"Target         : {target.label} "
                      f"{target_index+1} at {target.center}")
                if aim_info:
                    print(f"Advice         : {aim_info['advice']}")
                print(f"{'─'*40}")

        # ── 5. VISUALIZE ──────────────────────────────────────
        if show_mask:
            display = pig_det.get_mask_visual(frame)

        else:
            display = frame.copy()

            # Draw all target boxes (green=pig, red=TNT)
            display = pig_det.visualize(display, all_targets)

            # Draw slingshot crosshair
            display = sling_det.visualize(display, sling_pos)

            # Draw trajectory and aiming info
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

        # FPS counter
        cv2.putText(
            display, f"FPS: {fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 200, 255), 2
        )

        # Target info
        if target:
            target_text = (
                f"Target: {target.label} {target_index+1} "
                f"at {target.center}"
            )
            cv2.putText(
                display, target_text,
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )

        # TNT warning if TNT is targeted
        if target and target.label == "tnt":
            cv2.putText(
                display, "TARGET: TNT BOX",
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        # ── 7. SCALE AND SHOW ─────────────────────────────────
        h, w    = display.shape[:2]
        display = cv2.resize(
            display,
            (int(w * DISPLAY_SCALE),
             int(h * DISPLAY_SCALE))
        )
        cv2.imshow(WINDOW_NAME, display)

        # ── 8. KEYBOARD INPUT ─────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            logger.info("Quit.")
            break

        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOTS_DIR, f"frame_{ts}.png")
            cv2.imwrite(path, frame)
            logger.info(f"Screenshot saved: {path}")

        elif key == ord('d'):
            show_mask = not show_mask
            logger.info(f"Mask view: {show_mask}")

        elif key == ord('t'):
            show_traj = not show_traj
            logger.info(f"Trajectory: {show_traj}")

        elif key == ord('n'):
            if all_targets:
                target_index = (target_index + 1) % len(all_targets)
                new_target   = all_targets[target_index]
                logger.info(
                    f"Target changed to: "
                    f"{new_target.label} {target_index+1} "
                    f"at {new_target.center}"
                )

    # ── CLEANUP ───────────────────────────────────────────────
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    avg_fps    = frame_count / total_time
    logger.info(
        f"Session ended. "
        f"{frame_count} frames in "
        f"{total_time:.1f}s "
        f"(avg {avg_fps:.1f} FPS)"
    )


if __name__ == "__main__":
    run()