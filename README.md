# 🐦 Bird Watcher — OpenClaw Skill

Real-time bird detection using YOLOv11 + Moondream VLM for species identification. Streams a live annotated video feed to any device on your local network.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Grant camera permission (macOS — run once)
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"

# Start live stream
python3 bird_watcher_stream.py
```

Open `http://YOUR_IP:8888` on your phone, tablet, or AirPlay to TV.

## Features

- 🎥 **Live video stream** at full camera fps with YOLO bounding boxes
- 🧠 **Species identification** via Moondream VLM on detected birds
- 💾 **Auto-saves** detection frames (original + annotated)
- 📊 **HUD overlay** with bird count, fps, detection log
- 📱 **AirPlay compatible** — view on any device on your network
- 🔗 **OpenClaw integration** — wildlife census logging, BirdNET audio, Telegram alerts

## YOLO Models

| Model | FPS (M1) | Accuracy | Command |
|-------|----------|----------|---------|
| Nano | ~15-25 | Good | `--model yolo11n.pt` |
| **Small** | **~10-15** | **Better** | **`--model yolo11s.pt` (default)** |
| Medium | ~5-8 | Great | `--model yolo11m.pt` |

## Architecture

Camera and YOLO run in separate threads — video is always smooth, YOLO boxes update independently.

See [SKILL.md](SKILL.md) for full documentation.

## License

MIT — Created by Victor & Mark as an [OpenClaw](https://openclaw.ai) skill.
