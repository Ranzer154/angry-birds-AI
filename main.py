# main.py

import cv2
import os
import time
from datetime import datetime

from src.vision.capture import ScreenCapture
from src.vision.detector import PigDetector
from src.utils.window_finder import get_angry_birds_region
from src.utils.config import DISPLAY_SCALE, WINDOW_NAME, SCREENSHOTS_DIR
from src.utils.logger import get_logger

logger = get_logger("main")


def run():

    logger.info("Angry Birds AI — Milestone 1 Starting")
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    region   = get_angry_birds_region()
    capture  = ScreenCapture(region=region)
    detector = PigDetector()

    show_mask   = False
    frame_count = 0
    start_time  = time.time()

    logger.info("Running. Keys: Q=quit  S=save  D=mask  P=print")

    while True:
        frame      = capture.capture()
        frame_count += 1
        detections = detector.detect(frame)

        # Print coordinates every 30 frames
        if frame_count % 30 == 0 and detections:
            print(f"\n─── Frame {frame_count} | {len(detections)} pig(s) ───")
            for i, det in enumerate(detections):
                print(f"  Pig {i+1}: {det}")

        # Draw boxes or show mask
        if show_mask:
            display = detector.get_mask_visual(frame)
        else:
            display = detector.visualize(frame, detections)

        # FPS overlay
        elapsed = time.time() - start_time
        fps     = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            display, f"FPS: {fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 200, 255), 2
        )

        # Scale and show
        h, w    = display.shape[:2]
        display = cv2.resize(display, (int(w * DISPLAY_SCALE),
                                       int(h * DISPLAY_SCALE)))
        cv2.imshow(WINDOW_NAME, display)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOTS_DIR, f"frame_{ts}.png")
            cv2.imwrite(path, frame)
            logger.info(f"Saved: {path}")
        elif key == ord('d'):
            show_mask = not show_mask
            logger.info(f"Mask: {show_mask}")
        elif key == ord('p'):
            print(f"\n─── Manual print | Frame {frame_count} ───")
            for i, det in enumerate(detections):
                print(f"  Pig {i+1}: {det.to_dict()}")

    cv2.destroyAllWindows()
    logger.info(f"Done. {frame_count} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    run()