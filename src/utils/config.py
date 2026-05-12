# src/utils/config.py

# ── CAPTURE REGION ────────────────────────────────────────────
# Pixel coordinates of your game window.
# We will update these after running calibrate.py
CAPTURE_REGION = {
    "top":    50,
    "left":   9,
    "width":  941,
    "height": 965
}

# ── PIG COLOR DETECTION ───────────────────────────────────────
# HSV color range for pig-green color
# Format: (Hue, Saturation, Value)
PIG_HSV_LOWER = (27, 190, 60)
PIG_HSV_UPPER = (62, 255, 255)

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