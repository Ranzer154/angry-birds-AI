# 🐦 Angry Birds AI Bot

An AI-powered bot that plays Angry Birds 2 using 
Computer Vision and Reinforcement Learning.

## 🎯 Current Status
✅ Milestone 1 — Real-time pig detection working at 20+ FPS

## 📸 Demo
The bot detects pigs in real-time and draws bounding boxes around them.

## 🛠️ Tech Stack
- Python 3.11
- OpenCV — computer vision
- mss — screen capture
- pyautogui — mouse automation
- YOLOv8 — coming in Phase 2
- PyTorch + Stable-Baselines3 — coming in Phase 6

## ⚙️ Setup

### 1. Clone the repository
git clone https://github.com/Ranzer154/angry-birds-AI.git
cd angry-birds-AI

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install opencv-python numpy mss pyautogui Pillow matplotlib

### 4. Calibrate for your screen
python calibrate.py

### 5. Run the bot
python main.py

## ⌨️ Controls
| Key | Action |
|-----|--------|
| Q | Quit |
| S | Save screenshot |
| D | Toggle debug mask |
| P | Print pig coordinates |

## 🗺️ Roadmap
- [x] Phase 1 — Screen capture + pig detection
- [ ] Phase 2 — Custom YOLOv8 model
- [ ] Phase 3 — Trajectory prediction
- [ ] Phase 4 — Mouse automation
- [ ] Phase 5 — Heuristic AI
- [ ] Phase 6 — Reinforcement Learning agent

## 📁 Project Structure
angrybirds-ai/
├── src/
│   ├── vision/       # screen capture + detection
│   ├── aiming/       # physics + trajectory
│   ├── automation/   # mouse control
│   ├── rl/           # reinforcement learning
│   └── utils/        # config + logging
├── datasets/         # training data
├── models/           # trained weights
├── screenshots/      # saved frames
├── calibrate.py      # setup tool
└── main.py           # entry point

## 🧠 How It Works
1. Captures game screen at 20+ FPS using mss
2. Converts BGR image to HSV color space
3. Creates binary mask for pig-green color range
4. Finds contours in the mask
5. Filters by area to remove noise
6. Draws bounding boxes around detected pigs
7. Prints real-time coordinates to terminal