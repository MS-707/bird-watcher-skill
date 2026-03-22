#!/usr/bin/env python3
"""
Bird Watcher — Camera thread module.
Captures frames at full native fps, composites YOLO overlay, produces MJPEG frames.
"""

import cv2
import time
import logging
import threading
import numpy as np
from datetime import datetime

logger = logging.getLogger("bird-watcher")


def camera_thread(config, shared_state):
    """
    Capture frames at full camera speed. Never blocked by YOLO.

    Parameters
    ----------
    config : argparse.Namespace
        Configuration from config.get_config().
    shared_state : dict
        Thread-safe shared state dictionary with locks and mutable containers.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera: %dx%d @ %sfps", w, h, cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    fps_start = time.time()
    stats = shared_state["stats"]

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

        with shared_state["current_frame_lock"]:
            shared_state["current_frame"] = frame

        # Compose: raw frame + latest YOLO boxes overlay
        display = frame.copy()

        with shared_state["boxes_lock"]:
            active_boxes = shared_state["current_boxes"].copy()
            box_age = time.time() - shared_state["boxes_timestamp"]

        persist = config.persist
        min_bird_size = config.min_bird_size

        # Draw boxes if fresh enough
        if active_boxes and box_age < persist:
            fade = max(0.3, 1.0 - (box_age / persist))
            for x1, y1, x2, y2, conf in active_boxes:
                color = (0, int(255 * fade), int(100 * fade))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # Glow
                overlay = display.copy()
                cv2.rectangle(overlay, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 3)
                display = cv2.addWeighted(overlay, 0.2, display, 0.8, 0)

                bird_w = x2 - x1
                bird_h = y2 - y1
                label = f"Bird {conf:.0%}"
                sp = stats["last_species"]
                if (sp and sp not in ("Bird", "—", "")
                        and "unknown" not in sp.lower()
                        and "not sure" not in sp.lower()
                        and "can't" not in sp.lower()):
                    label = f"{sp.split(',')[0].split('.')[0].strip()[:22]} {conf:.0%}"
                if bird_w < min_bird_size or bird_h < min_bird_size:
                    label = f"Bird {conf:.0%} (far)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(display, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            stats["bird_count"] = len(active_boxes)
        elif box_age >= persist:
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
        with shared_state["stream_lock"]:
            shared_state["stream_jpeg"] = jpeg.tobytes()

    cap.release()
