# calibrate.py

import cv2
import numpy as np
import time
import pyautogui
from src.vision.capture import ScreenCapture


def step1_find_window():
    print("\n" + "="*50)
    print("STEP 1: Finding your game window")
    print("="*50)
    print("\nOpen Angry Birds so it is visible on screen.")
    print("Move mouse to TOP-LEFT corner of the game window.")
    print("Waiting 5 seconds...")
    time.sleep(5)

    x1, y1 = pyautogui.position()
    print(f"Top-left: ({x1}, {y1})")

    print("\nNow move mouse to BOTTOM-RIGHT corner.")
    print("Waiting 5 seconds...")
    time.sleep(5)

    x2, y2 = pyautogui.position()
    print(f"Bottom-right: ({x2}, {y2})")

    print(f"\nCopy this into src/utils/config.py:")
    print(f"CAPTURE_REGION = {{")
    print(f'    "top":    {y1},')
    print(f'    "left":   {x1},')
    print(f'    "width":  {x2 - x1},')
    print(f'    "height": {y2 - y1}')
    print(f"}}")


def step2_tune_hsv():
    print("\n" + "="*50)
    print("STEP 2: HSV Color Tuning")
    print("="*50)
    print("Adjust sliders until PIGS appear WHITE. Press Q when done.\n")

    cap = ScreenCapture()
    win = "HSV Tuner — Q to quit"
    cv2.namedWindow(win)

    cv2.createTrackbar("H lower", win, 30,  179, lambda x: None)
    cv2.createTrackbar("H upper", win, 85,  179, lambda x: None)
    cv2.createTrackbar("S lower", win, 50,  255, lambda x: None)
    cv2.createTrackbar("S upper", win, 255, 255, lambda x: None)
    cv2.createTrackbar("V lower", win, 50,  255, lambda x: None)
    cv2.createTrackbar("V upper", win, 255, 255, lambda x: None)

    h_lo = h_hi = s_lo = s_hi = v_lo = v_hi = 0

    while True:
        frame = cap.capture()
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_lo = cv2.getTrackbarPos("H lower", win)
        h_hi = cv2.getTrackbarPos("H upper", win)
        s_lo = cv2.getTrackbarPos("S lower", win)
        s_hi = cv2.getTrackbarPos("S upper", win)
        v_lo = cv2.getTrackbarPos("V lower", win)
        v_hi = cv2.getTrackbarPos("V upper", win)

        lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([
            cv2.resize(frame,    (640, 360)),
            cv2.resize(mask_bgr, (640, 360))
        ])
        cv2.imshow(win, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\nCopy these into src/utils/config.py:")
    print(f"PIG_HSV_LOWER = ({h_lo}, {s_lo}, {v_lo})")
    print(f"PIG_HSV_UPPER = ({h_hi}, {s_hi}, {v_hi})")


if __name__ == "__main__":
    print("Angry Birds AI — Calibration Tool")
    choice = input("\nRun [1] window finder  [2] HSV tuner  [3] both → ")

    if choice == "1":
        step1_find_window()
    elif choice == "2":
        step2_tune_hsv()
    else:
        step1_find_window()
        step2_tune_hsv()

    print("\nCalibration done. Now run: python main.py")