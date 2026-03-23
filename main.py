#!/usr/bin/env python3
"""
Bird Watcher Live Stream v3 — Decoupled camera + YOLO architecture.
Camera runs at full native fps. YOLO runs in a separate thread.
Bounding boxes overlay onto the smooth feed without slowing it down.

Usage: python3 main.py [--port 8888] [--model yolo11s.pt] [--confidence 0.15] [--persist 3]
"""

import ipaddress
import os
import signal
import socket
import logging
import sys
import threading

import cv2
from ultralytics import YOLO

from config import get_config, setup_logging, make_stats, stats_lock
from camera import camera_thread
from detector import yolo_thread
from species_id import verify_moondream
from storage import ensure_directories
from stream_server import app, init_server

logger = logging.getLogger("bird-watcher")


def build_shared_state():
    """Create the thread-safe shared state dictionary."""
    return {
        "current_frame": None,
        "current_frame_lock": threading.Lock(),
        "current_boxes": [],
        "boxes_lock": threading.Lock(),
        "boxes_timestamp": 0,
        "stream_jpeg": None,
        "stream_lock": threading.Lock(),
        "stats": make_stats(),
        "stats_lock": stats_lock,
    }


def get_local_ip():
    """Detect the local network IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def main():
    setup_logging()

    config = get_config(mode="stream")
    ensure_directories(config)

    # Load YOLO model
    logger.info("Loading %s...", config.model)
    model = YOLO(config.model)
    logger.info("Model loaded!")

    # Check Moondream availability
    moondream_available = verify_moondream(config.moondream_url)

    # Build shared state
    shared_state = build_shared_state()

    # Initialize Flask server
    init_server(config, shared_state)

    local_ip = get_local_ip()

    # Check if IP is public (non-RFC1918)
    try:
        ip_obj = ipaddress.ip_address(local_ip)
        if not ip_obj.is_private:
            print("⚠️  WARNING: Your IP appears to be public. "
                  "The camera stream may be accessible from the internet.")
    except ValueError:
        pass

    # Startup banner (user-facing HUD — intentionally print, not logger)
    print(f"\n{'=' * 60}")
    print(f"🐦 Bird Watcher Live Stream v3 — Decoupled Architecture")
    print(f"   Camera: Full native fps (decoupled from YOLO)")
    print(f"   YOLO: {config.model} (runs independently)")
    print(f"   Moondream: Species ID on detections (5s cooldown)")
    print(f"   Confidence: {config.confidence}")
    print(f"   Box persistence: {config.persist}s")
    print(f"")
    print(f"   Local:   http://localhost:{config.port}")
    print(f"   Network: http://{local_ip}:{config.port}")
    print(f"   Moondream: {'Connected ✅' if moondream_available else 'Not available (no species ID)'}")
    print(f"   Max viewers: {config.max_concurrent_viewers}")
    print(f"   Max saved frames: {config.max_detection_files}")
    print(f"")
    print(f"   🔐 Stream URL (share this):")
    print(f"   http://{local_ip}:{config.port}?token={config.stream_token}")
    print(f"")
    print(f"   AirPlay: Open the URL on iPhone → AirPlay to TV")
    print(f"{'=' * 60}\n")

    # Graceful shutdown handler
    def _shutdown_handler(signum, frame):
        print("\n🐦 Shutting down Bird Watcher...")
        # Release camera if possible
        try:
            cap = cv2.VideoCapture(0)
            cap.release()
        except Exception:
            pass
        print("Camera released. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)

    # Start threads
    t_cam = threading.Thread(target=camera_thread, args=(config, shared_state), daemon=True)
    t_yolo = threading.Thread(target=yolo_thread, args=(model, config, shared_state, moondream_available), daemon=True)
    t_cam.start()
    t_yolo.start()

    app.run(host='0.0.0.0', port=config.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
