#!/usr/bin/env python3
"""
Bird Watcher — YOLO detector thread module.
Runs YOLO detection on latest frames independently of camera speed.
"""

import os
import time
import logging
import threading

import cv2

from species_id import moondream_identify

logger = logging.getLogger("bird-watcher")


def yolo_thread(model, config, shared_state, moondream_available):
    """
    Run YOLO detection on latest frames. Independent of camera speed.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded YOLO model instance.
    config : argparse.Namespace
        Configuration namespace.
    shared_state : dict
        Thread-safe shared state dictionary.
    moondream_available : bool
        Whether Moondream VLM is reachable.
    """
    stats = shared_state["stats"]
    s_lock = shared_state.get("stats_lock")
    moondream_cooldown = 0
    frame_count = 0
    fps_start = time.time()

    while True:
        with shared_state["current_frame_lock"]:
            frame = shared_state["current_frame"]

        if frame is None:
            time.sleep(0.01)
            continue

        # Run YOLO
        results = model(frame, verbose=False, conf=config.confidence)
        birds = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == config.bird_class_id:
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
            with shared_state["boxes_lock"]:
                shared_state["current_boxes"] = birds
                shared_state["boxes_timestamp"] = time.time()

            if s_lock:
                with s_lock:
                    stats["total_detections"] += 1
            else:
                stats["total_detections"] += 1

            if not getattr(config, 'no_save', False):
                # Auto-cleanup old detections if over limit
                _cleanup_detections(config.detections_dir, config.max_detection_files)

                # Save detection frames
                _save_detection_frames(frame, birds, config.detections_dir)

            # Moondream species ID (with cooldown, only for birds big enough)
            if time.time() > moondream_cooldown:
                biggest = max(birds, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                bw = biggest[2] - biggest[0]
                bh = biggest[3] - biggest[1]
                if bw >= config.min_bird_size and bh >= config.min_bird_size:
                    threading.Thread(
                        target=moondream_identify,
                        args=(frame.copy(), biggest[:4], config, stats, moondream_available, s_lock),
                        daemon=True,
                    ).start()
                    moondream_cooldown = time.time() + 5

        time.sleep(0.01)  # Yield to other threads


def _cleanup_detections(detections_dir, max_files):
    """Remove oldest detection files if count exceeds max_files."""
    try:
        det_files = sorted(
            [f for f in os.listdir(detections_dir) if f.endswith('.jpg')],
            key=lambda f: os.path.getmtime(os.path.join(detections_dir, f)),
        )
        while len(det_files) > max_files:
            oldest = det_files.pop(0)
            os.remove(os.path.join(detections_dir, oldest))
    except OSError as exc:
        logger.warning("Detection cleanup error: %s", exc)


def _save_detection_frames(frame, birds, detections_dir):
    """Save original and annotated detection frames to disk."""
    from datetime import datetime

    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Original frame
    orig_path = os.path.join(detections_dir, f"orig_{ts_str}.jpg")
    cv2.imwrite(orig_path, frame)

    # Annotated version
    det_frame = frame.copy()
    for x1, y1, x2, y2, conf in birds:
        cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        label = f"Bird {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(det_frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 100), -1)
        cv2.putText(det_frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    det_path = os.path.join(detections_dir, f"det_{ts_str}.jpg")
    cv2.imwrite(det_path, det_frame)
