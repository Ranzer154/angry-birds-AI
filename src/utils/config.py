# src/utils/config.py

# ── CAPTURE REGION ────────────────────────────────────────────
# Pixel coordinates of your game window.
# We will update these after running calibrate.py
CAPTURE_REGION = {
    "top":    100,
    "left":   100,
    "width":  1280,
    "height": 720
}

# ── PIG COLOR DETECTION ───────────────────────────────────────
# HSV color range for pig-green color
# Format: (Hue, Saturation, Value)
PIG_HSV_LOWER = (30, 50, 50)
PIG_HSV_UPPER = (85, 255, 255)

# Minimum pixel area to count as a pig (filters out noise)
MIN_PIG_AREA = 300

# ── DISPLAY ───────────────────────────────────────────────────
# Scale the debug window to 75% so it fits your screen
DISPLAY_SCALE = 0.75

# Title of the OpenCV debug window
WINDOW_NAME = "Angry Birds AI — Debug View"

# ── OUTPUT PATHS ──────────────────────────────────────────────
SCREENSHOTS_DIR = "screenshots"
DATASETS_DIR    = "datasets"
MODELS_DIR      = "models"