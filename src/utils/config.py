# src/utils/config.py

CAPTURE_REGION = {
    "top":    45,
    "left":   15,
    "width":  930,
    "height": 965
}

# Pig detection — tuned from real samples
PIG_HSV_LOWER = (25, 170, 200)
PIG_HSV_UPPER = (61, 255, 255)

MIN_PIG_AREA = 500

DISPLAY_SCALE = 0.8

WINDOW_NAME = "Angry Birds AI — Debug View"

SCREENSHOTS_DIR = "screenshots"
DATASETS_DIR    = "datasets"
MODELS_DIR      = "models"