---
name: bird-watcher
description: "Live bird detection with YOLO + VLM species ID. Streams a real-time annotated video feed to any device on your network. Detects birds with bounding boxes, identifies species via Moondream VLM, records audio with BirdNET. Use when: user asks to watch for birds, detect wildlife, set up a bird camera, or run a wildlife monitoring stream."
---

# Bird Watcher — Live YOLO Bird Detection + VLM Species ID

Real-time bird detection system using YOLOv11 for fast visual detection and Moondream VLM for species identification. Streams an annotated live video feed to any device on your local network via MJPEG.

## What It Does

- **Live video stream** with YOLO bounding boxes at full camera fps
- **Species identification** via Moondream VLM when a bird is detected
- **Auto-saves** original + annotated frames of every detection
- **Wildlife census logging** for tracking species over time
- **BirdNET audio** detection (optional, requires microphone)
- **AirPlay compatible** — open the URL on iPhone, cast to Apple TV

## Architecture

```
Camera (30fps) → Camera Thread (smooth streaming)
                      ↓ (shares frames)
                 YOLO Thread (independent, ~10-15fps)
                      ↓ (bird detected?)
                 Moondream VLM (species ID, 5s cooldown)
                      ↓
                 Bounding boxes overlaid on stream
                 + Detection frames saved to disk
```

Camera and YOLO run in separate threads — the video feed is always smooth regardless of YOLO processing speed.

## Requirements

### Hardware
- macOS with a webcam (MacBook built-in or USB camera)
- Recommended: Apple Silicon (M1/M2/M3/M4) for best YOLO performance
- Intel Macs work but expect lower fps

### Software
- Python 3.10+
- Moondream Station running on localhost:2020 (optional, for species ID)

## Agent Installation Guide

When a user asks to set up bird watching, follow these steps in order. Each step has a verification check — do NOT proceed until the check passes.

### Step 1: Check Python version
```bash
python3 --version
```
**Requires:** Python 3.10+. If not installed, tell the user to install via `brew install python3` or from python.org.

### Step 2: Install dependencies
```bash
pip3 install --break-system-packages ultralytics opencv-python flask requests numpy
```
**Verify:**
```bash
python3 -c "import cv2, flask, ultralytics, requests; print('All dependencies OK')"
```
If `--break-system-packages` fails, try without it. If that fails, try `pip3 install --user`. On some systems you may need a virtual environment:
```bash
python3 -m venv ~/.openclaw/skills/bird-watcher/venv
source ~/.openclaw/skills/bird-watcher/venv/bin/activate
pip install ultralytics opencv-python flask requests numpy
```

### Step 3: Download YOLO model
The model auto-downloads on first run, but you can pre-download:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); print('Model downloaded')"
```
**Verify:** File `yolo11s.pt` exists in the working directory.

### Step 4: Camera permissions (macOS — CRITICAL)
This is the most common failure point. Python MUST have camera access granted by the user.

**Tell the user:** "I need you to run one command in Terminal to grant camera permission. Open Terminal and paste this:"
```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"
```
macOS will show a dialog asking to allow camera access. The user MUST click **Allow**.

**Verify:** The command above prints `Camera: True`. If it prints `False`, camera access was denied. The user needs to go to System Settings → Privacy & Security → Camera and enable it for Terminal or Python.

**IMPORTANT:** This command must be run interactively in Terminal by the user, NOT via exec/nohup/background. macOS only shows the permission dialog for interactive foreground processes.

### Step 5: Moondream Station (optional but recommended)
Moondream provides species identification. Without it, you get bounding boxes but no species names.

**Check if running:**
```bash
curl -s http://localhost:2020/health
```
If it returns `{"status":"ok"}`, Moondream is ready. If not, the bird watcher still works — it just won't identify species.

### Step 6: Firewall (for network streaming)
If the user wants to view the stream on their phone/TV:

**Tell the user:** "To view the stream on other devices, you'll need to either disable the macOS firewall temporarily or add Python to the allowed apps. Go to System Settings → Network → Firewall."

**Verify:**
```bash
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```
If enabled, the stream will only work on `localhost` unless the user adjusts it.

### Step 7: Start the stream
The stream script MUST be run interactively in Terminal (for camera access):

**Tell the user:** "Run this in your Terminal window:"
```bash
cd ~/.openclaw/skills/bird-watcher && python3 bird_watcher_stream.py
```

Then provide them the URL: `http://<LOCAL_IP>:8888`

To find the local IP:
```bash
ipconfig getifaddr en0
```

### Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | `pip3 install --break-system-packages opencv-python` |
| `ModuleNotFoundError: No module named 'ultralytics'` | YOLO not installed | `pip3 install --break-system-packages ultralytics` |
| `Camera: False` or black screen | No camera permission | User must run the camera check command interactively in Terminal and click Allow |
| `OpenCV: not authorized to capture video` | Running in background | Must run the stream script in a foreground Terminal window, not via nohup/exec |
| Stream works on localhost but not phone | macOS firewall blocking | User disables firewall or adds Python to allowed apps |
| `Address already in use` on port 8888 | Previous instance running | `kill $(lsof -ti:8888)` or use `--port 9999` |
| Low fps (<5) | Using large YOLO model | Switch to `--model yolo11n.pt` for speed |
| No species identification | Moondream not running | Install and start Moondream Station on localhost:2020 |
| `No module named 'torch'` | PyTorch not installed | `pip3 install --break-system-packages torch torchvision` |

## Manual Installation (for users)

```bash
# Clone the skill
git clone https://github.com/MS-707/bird-watcher-skill.git
cd bird-watcher-skill

# Install dependencies
pip3 install --break-system-packages -r requirements.txt

# Grant camera permission (run in Terminal, click Allow)
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"

# Start the stream
python3 bird_watcher_stream.py
```

### Firewall

If you want to view the stream from other devices (phone, TV), either:
- Temporarily disable the macOS firewall, OR
- Add Python to the firewall allow list in System Settings → Network → Firewall → Options

## Usage

### Start the live stream
```bash
python3 bird_watcher_stream.py
```

Opens a web server at `http://YOUR_IP:8888`. Open this URL on any device on your network.

### Options
```bash
python3 bird_watcher_stream.py --port 9999           # Custom port
python3 bird_watcher_stream.py --model yolo11n.pt     # Faster, less accurate
python3 bird_watcher_stream.py --model yolo11m.pt     # Slower, more accurate
python3 bird_watcher_stream.py --confidence 0.20      # Adjust detection threshold
python3 bird_watcher_stream.py --persist 5             # Bounding box persistence (seconds)
```

### YOLO Model Selection

| Model | FPS (M1) | Accuracy | Best For |
|-------|----------|----------|----------|
| `yolo11n.pt` | ~15-25 | Good | Maximum smoothness |
| `yolo11s.pt` | ~10-15 | Better | **Recommended** — good balance |
| `yolo11m.pt` | ~5-8 | Great | Best detection, still usable |
| `yolo11l.pt` | ~2-4 | Excellent | Maximum accuracy |

### Batch detection (no stream)
```bash
python3 bird_watcher_batch.py --duration 300 --interval 10
```

Captures frames every N seconds, runs YOLO + Moondream, saves detections. Good for unattended monitoring.

## Detection Output

Detections are saved to `./detections/`:
- `orig_YYYYMMDD_HHMMSS.jpg` — original clean frame
- `det_YYYYMMDD_HHMMSS.jpg` — annotated frame with bounding boxes
- `session_YYYYMMDD_HHMMSS.json` — session summary with all detections

## Integration with OpenClaw

### Wildlife Census
If the `wildlife-census` skill is installed, detections are automatically logged:
```bash
~/.openclaw/skills/wildlife-census/census.sh log "Scrub Jay" 1 "BirdWatcher: detected at feeder"
```

### BirdNET Audio
If the `birdnet-audio` skill is installed, audio detection runs in parallel with video.

### Telegram Alerts
Configure your OpenClaw agent to send detection photos to Telegram when a bird is spotted.

## How It Works

1. **Camera Thread** captures frames at native camera fps (typically 30fps on MacBook)
2. **YOLO Thread** pulls the latest frame and runs YOLOv11 bird detection independently
3. When YOLO detects a bird (class 14 in COCO), it stores bounding box coordinates
4. **Camera Thread** overlays the latest bounding boxes onto the smooth video feed
5. Boxes persist for N seconds (default 3) so fast fly-throughs are visible
6. **Moondream VLM** runs on detected bird crops for species identification (5s cooldown)
7. All detections saved to disk with original + annotated frames

## Troubleshooting

### Camera not working
- Run the permission command above in Terminal
- Check System Settings → Privacy & Security → Camera
- Make sure no other app (FaceTime, Zoom) has the camera open

### Black screen on phone
- Make sure phone is on same WiFi as the Mac
- Check macOS firewall settings
- Try `http://localhost:8888` on the Mac first to verify the stream works

### Low fps
- Use a smaller YOLO model (`yolo11n.pt`)
- Close other apps using GPU
- Reduce camera resolution in the script

### No birds detected
- Lower the confidence threshold (`--confidence 0.10`)
- Make sure the camera has a clear view of the feeder/area
- Birds may be too small/far — try a closer camera position

## Credits

- [YOLOv11](https://github.com/ultralytics/ultralytics) by Ultralytics
- [Moondream](https://moondream.ai/) for local VLM inference
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) by Cornell Lab of Ornithology
- Created by Victor & Mark as an OpenClaw skill
