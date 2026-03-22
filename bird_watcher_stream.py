#!/usr/bin/env python3
"""
Bird Watcher Live Stream v3 — Decoupled camera + YOLO architecture
Camera runs at full native fps. YOLO runs in a separate thread.
Bounding boxes overlay onto the smooth feed without slowing it down.

Usage: python3 live_stream_v3.py [--port 8888] [--model yolo11s.pt]
"""

import cv2
import time
import sys
import os
import base64
import requests
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

# Config
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8888
MODEL_NAME = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "yolo11s.pt"
BIRD_CLASS_ID = 14
CONFIDENCE_THRESHOLD = 0.15
DETECTION_PERSIST_SECONDS = 3
MIN_BIRD_SIZE = 50  # Minimum pixel width/height to trigger Moondream species ID
MOONDREAM_URL = "http://localhost:2020"
MAX_DETECTION_FILES = 500  # Auto-cleanup after this many saved frames
MAX_CONCURRENT_VIEWERS = 5

# Paths — use relative/expanduser, never hardcoded usernames
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTIONS_DIR = os.path.join(SKILL_DIR, "detections")
CENSUS_SCRIPT = os.path.expanduser("~/.openclaw/skills/wildlife-census/census.sh")

# Auth — generate a random token on startup for stream access
import secrets
STREAM_TOKEN = os.environ.get("BIRDWATCH_TOKEN", secrets.token_urlsafe(16))
active_viewers = 0
viewers_lock = threading.Lock()

os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Load YOLO
print(f"Loading {MODEL_NAME}...")
model = YOLO(MODEL_NAME)
print("Model loaded!")

app = Flask(__name__)

# Shared state — thread-safe via locks
current_frame = None
current_frame_lock = threading.Lock()

current_boxes = []  # Latest YOLO detections
boxes_lock = threading.Lock()
boxes_timestamp = 0

stream_jpeg = None
stream_lock = threading.Lock()

# Stats
stats = {
    "camera_fps": 0,
    "yolo_fps": 0,
    "bird_count": 0,
    "total_detections": 0,
    "last_species": "Bird",
    "detection_log": [],
}


def verify_moondream():
    """Check if Moondream Station is actually running."""
    try:
        resp = requests.get(f"{MOONDREAM_URL}/health", timeout=3)
        if resp.ok and resp.json().get("server") == "moondream-station":
            return True
    except:
        pass
    return False

moondream_available = False  # Set during startup


def moondream_identify(img, bbox):
    """Species ID via Moondream VLM."""
    if not moondream_available:
        return
    try:
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        pad = 0.25
        crop = img[
            max(0, int(y1 - (y2-y1)*pad)):min(h, int(y2 + (y2-y1)*pad)),
            max(0, int(x1 - (x2-x1)*pad)):min(w, int(x2 + (x2-x1)*pad))
        ]
        if crop.size == 0:
            return
        _, buffer = cv2.imencode(".jpg", crop)
        img_b64 = base64.b64encode(buffer).decode()
        resp = requests.post(
            f"{MOONDREAM_URL}/caption",
            json={"image": img_b64, "prompt": "Identify this bird species. Just the name and one brief detail."},
            timeout=10
        )
        if resp.ok:
            data = resp.json()
            species = data.get("caption", data.get("result", "")).strip()
            # Only update species if Moondream gave a real answer
            bad_answers = ["unknown", "not sure", "can't tell", "i can't", "cannot", "unclear", "hard to", "difficult to", "image", "photo", "picture"]
            if species and not any(bad in species.lower() for bad in bad_answers):
                stats["last_species"] = species
            # If Moondream couldn't ID it, keep the default "Bird"
            stats["detection_log"].append({"time": datetime.now().strftime("%H:%M:%S"), "species": species})
            if len(stats["detection_log"]) > 100:
                stats["detection_log"].pop(0)
            
            # Log to wildlife census (only if script exists)
            clean = species.split(",")[0].split(".")[0].strip()
            if os.path.exists(CENSUS_SCRIPT):
                try:
                    import subprocess
                    subprocess.run([CENSUS_SCRIPT, "log", clean, "1", f"BirdWatcher: {species[:80]}"],
                                 capture_output=True, timeout=10)
                except:
                    pass
    except:
        pass


def camera_thread():
    """Captures frames at full camera speed. Never blocked by YOLO."""
    global current_frame, stream_jpeg

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS)}fps")

    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue

        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            stats["camera_fps"] = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        with current_frame_lock:
            current_frame = frame

        # Compose: raw frame + latest YOLO boxes overlay
        display = frame.copy()

        with boxes_lock:
            active_boxes = current_boxes.copy()
            box_age = time.time() - boxes_timestamp

        # Draw boxes if fresh enough
        if active_boxes and box_age < DETECTION_PERSIST_SECONDS:
            fade = max(0.3, 1.0 - (box_age / DETECTION_PERSIST_SECONDS))
            for x1, y1, x2, y2, conf in active_boxes:
                color = (0, int(255 * fade), int(100 * fade))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # Glow
                overlay = display.copy()
                cv2.rectangle(overlay, (x1-1, y1-1), (x2+1, y2+1), color, 3)
                display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)

                bird_w = x2 - x1
                bird_h = y2 - y1
                label = f"Bird {conf:.0%}"
                sp = stats["last_species"]
                if sp and sp not in ("Bird", "—", "") and "unknown" not in sp.lower() and "not sure" not in sp.lower() and "can't" not in sp.lower():
                    label = f"{sp.split(',')[0].split('.')[0].strip()[:22]} {conf:.0%}"
                if bird_w < MIN_BIRD_SIZE or bird_h < MIN_BIRD_SIZE:
                    label = f"Bird {conf:.0%} (far)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(display, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            stats["bird_count"] = len(active_boxes)
        elif box_age >= DETECTION_PERSIST_SECONDS:
            stats["bird_count"] = 0

        # HUD
        cam_fps = stats["camera_fps"]
        yolo_fps = stats["yolo_fps"]
        cv2.putText(display, f"Birds: {stats['bird_count']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        cv2.putText(display, f"cam:{cam_fps:.0f} yolo:{yolo_fps:.1f}fps", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 80), 1)

        ts = datetime.now().strftime("%H:%M:%S")
        status = f"Bird Watcher v3 | {ts} | Det: {stats['total_detections']}"
        sp = stats["last_species"]
        if sp != "—":
            status += f" | {sp.split(',')[0][:20]}"
        cv2.putText(display, status, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)

        _, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with stream_lock:
            stream_jpeg = jpeg.tobytes()

    cap.release()


# Make these accessible in camera_thread
current_frame = None


def yolo_thread():
    """Runs YOLO detection on latest frames. Independent of camera speed."""
    global current_boxes, boxes_timestamp

    moondream_cooldown = 0
    frame_count = 0
    fps_start = time.time()

    while True:
        with current_frame_lock:
            frame = current_frame

        if frame is None:
            time.sleep(0.01)
            continue

        # Run YOLO
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        birds = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == BIRD_CLASS_ID:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    birds.append((x1, y1, x2, y2, conf))

        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            stats["yolo_fps"] = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        if birds:
            with boxes_lock:
                current_boxes = birds
                boxes_timestamp = time.time()

            stats["total_detections"] += 1

            # Auto-cleanup old detections if over limit
            try:
                det_files = sorted(
                    [f for f in os.listdir(DETECTIONS_DIR) if f.endswith('.jpg')],
                    key=lambda f: os.path.getmtime(os.path.join(DETECTIONS_DIR, f))
                )
                while len(det_files) > MAX_DETECTION_FILES:
                    oldest = det_files.pop(0)
                    os.remove(os.path.join(DETECTIONS_DIR, oldest))
            except:
                pass

            # Save detection frames
            ts_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            orig_path = os.path.join(DETECTIONS_DIR, f"orig_{ts_str}.jpg")
            cv2.imwrite(orig_path, frame)

            # Annotated version
            det_frame = frame.copy()
            for x1, y1, x2, y2, conf in birds:
                cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                label = f"Bird {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(det_frame, (x1, y1-th-10), (x1+tw+6, y1), (0, 255, 100), -1)
                cv2.putText(det_frame, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            det_path = os.path.join(DETECTIONS_DIR, f"det_{ts_str}.jpg")
            cv2.imwrite(det_path, det_frame)

            # Moondream species ID (with cooldown, only for birds big enough to identify)
            if time.time() > moondream_cooldown:
                biggest = max(birds, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                bw = biggest[2] - biggest[0]
                bh = biggest[3] - biggest[1]
                if bw >= MIN_BIRD_SIZE and bh >= MIN_BIRD_SIZE:
                    threading.Thread(target=moondream_identify, args=(frame.copy(), biggest[:4]), daemon=True).start()
                    moondream_cooldown = time.time() + 5

        time.sleep(0.01)  # Yield to other threads


def generate_mjpeg():
    while True:
        with stream_lock:
            frame = stream_jpeg
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐦 Bird Watcher Live</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #000; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        img { max-width: 100%; max-height: 100vh; object-fit: contain; }
    </style>
</head>
<body>
    <img src="/feed" alt="Bird Watcher Live Feed">
</body>
</html>
"""


def check_auth():
    """Verify stream token in query params."""
    from flask import request, abort
    token = request.args.get('token', '')
    if token != STREAM_TOKEN:
        abort(403)


@app.route('/')
def index():
    from flask import request
    token = request.args.get('token', '')
    if token != STREAM_TOKEN:
        return '<html><body style="background:#000;color:#fff;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh"><div><h1>🐦 Bird Watcher</h1><p>Access token required. Add ?token=YOUR_TOKEN to the URL.</p></div></body></html>', 403
    page = HTML_PAGE.replace('/feed', f'/feed?token={STREAM_TOKEN}')
    return render_template_string(page)


@app.route('/feed')
def video_feed():
    global active_viewers
    from flask import request, abort
    token = request.args.get('token', '')
    if token != STREAM_TOKEN:
        abort(403)
    with viewers_lock:
        if active_viewers >= MAX_CONCURRENT_VIEWERS:
            abort(503)  # Too many viewers
        active_viewers += 1
    try:
        return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')
    finally:
        with viewers_lock:
            active_viewers = max(0, active_viewers - 1)


def main():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except:
        local_ip = "localhost"
    finally:
        s.close()

    global moondream_available
    moondream_available = verify_moondream()

    print(f"\n{'=' * 60}")
    print(f"🐦 Bird Watcher Live Stream v3 — Decoupled Architecture")
    print(f"   Camera: Full native fps (decoupled from YOLO)")
    print(f"   YOLO: {MODEL_NAME} (runs independently)")
    print(f"   Moondream: Species ID on detections (5s cooldown)")
    print(f"   Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"   Box persistence: {DETECTION_PERSIST_SECONDS}s")
    print(f"")
    print(f"   Local:   http://localhost:{PORT}")
    print(f"   Network: http://{local_ip}:{PORT}")
    print(f"   Moondream: {'Connected ✅' if moondream_available else 'Not available (no species ID)'}")
    print(f"   Max viewers: {MAX_CONCURRENT_VIEWERS}")
    print(f"   Max saved frames: {MAX_DETECTION_FILES}")
    print(f"")
    print(f"   🔐 Stream URL (share this):")
    print(f"   http://{local_ip}:{PORT}?token={STREAM_TOKEN}")
    print(f"")
    print(f"   AirPlay: Open the URL on iPhone → AirPlay to TV")
    print(f"{'=' * 60}\n")

    # Open camera on MAIN thread first (macOS requires this for permission)
    print("Opening camera on main thread...")
    _cap = cv2.VideoCapture(0)
    if not _cap.isOpened():
        print("ERROR: Cannot open camera! Grant permission in System Settings → Privacy → Camera")
        return
    print(f"Camera verified: {int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    _cap.release()
    import time
    time.sleep(0.5)  # Brief pause before re-opening in thread

    # Start threads
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_yolo = threading.Thread(target=yolo_thread, daemon=True)
    t_cam.start()
    t_yolo.start()

    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False)


if __name__ == '__main__':
    main()
