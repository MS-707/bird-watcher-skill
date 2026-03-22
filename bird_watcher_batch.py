#!/usr/bin/env python3
"""
Bird Watcher Batch v2 — YOLO + Moondream Hybrid Detection.
- YOLO for fast bird detection with bounding boxes
- Moondream VLM for species identification on detected birds
- BirdNET for audio species detection
- Saves annotated photos on detection

Usage: python3 bird_watcher_batch.py [--duration 1800] [--interval 8] [--model yolo11s.pt] [--confidence 0.15]
"""

import json
import logging
import os
import shutil
import subprocess
import time

import base64
import cv2
import numpy as np
import requests
from datetime import datetime
from ultralytics import YOLO

from config import get_config, setup_logging, CENSUS_SCRIPT

logger = logging.getLogger("bird-watcher")

SNAP_DIR = "/tmp/victor_vision"


def capture_frame(output_dir):
    """Capture via snap watcher (has camera permissions)."""
    os.makedirs(SNAP_DIR, exist_ok=True)
    ready_file = os.path.join(SNAP_DIR, "ready")
    try:
        os.remove(ready_file)
    except FileNotFoundError:
        pass

    with open(os.path.join(SNAP_DIR, "request"), "w") as f:
        f.write("bird detection")

    for _ in range(16):
        if os.path.exists(ready_file):
            with open(ready_file) as f:
                src_path = f.read().strip()
            os.remove(ready_file)
            if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
                shutil.copy2(src_path, dst_path)
                return dst_path
            return None
        time.sleep(0.5)
    return None


def yolo_detect_birds(model, image_path, confidence, bird_class_id):
    """Run YOLO on frame, return bird detections."""
    results = model(image_path, verbose=False, conf=confidence)
    birds = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == bird_class_id:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                birds.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                })
    return birds


def annotate_frame(image_path, birds, species_labels=None):
    """Draw bounding boxes and labels on the frame."""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    for i, bird in enumerate(birds):
        x1, y1, x2, y2 = bird["bbox"]
        conf = bird["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)

        label = f"Bird {conf:.0%}"
        if species_labels and i < len(species_labels):
            label = f"{species_labels[i]} {conf:.0%}"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 6, y1), (0, 255, 100), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, f"Bird Watcher v2 | {ts}", (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    cv2.putText(img, f"Birds: {len(birds)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

    annotated_path = image_path.replace("/frame_", "/annotated_")
    cv2.imwrite(annotated_path, img)
    return annotated_path


def moondream_identify_batch(image_path, bbox, moondream_url):
    """Crop bird region and ask Moondream for species ID."""
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown bird"

    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * 0.2)
    pad_y = int((y2 - y1) * 0.2)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    crop = img[cy1:cy2, cx1:cx2]

    _, buffer = cv2.imencode(".jpg", crop)
    img_b64 = base64.b64encode(buffer).decode()

    try:
        resp = requests.post(
            f"{moondream_url}/caption",
            json={
                "image": img_b64,
                "prompt": (
                    "What species of bird is this? Identify the species name. "
                    "Be specific and concise — just the species name and one brief "
                    "detail about its appearance."
                ),
            },
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            caption = data.get("caption", data.get("result", "Unknown bird"))
            return caption.strip()
    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.warning("Moondream error: %s", exc)
    except (ValueError, KeyError) as exc:
        logger.warning("Moondream response parsing error: %s", exc)
    return "Unknown bird"


def log_to_census(species, caption):
    """Log detection to wildlife census."""
    if not os.path.exists(CENSUS_SCRIPT):
        return
    try:
        subprocess.run(
            [CENSUS_SCRIPT, "log", species, "1", f"YOLO+Moondream: {caption[:100]}"],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Census logging failed: %s", exc)


def run_birdnet_async():
    """Start BirdNET listening in background."""
    birdnet_script = os.path.expanduser("~/.openclaw/skills/birdnet-audio/birdnet.sh")
    if os.path.exists(birdnet_script):
        logger.info("BirdNET recording (60s)...")
        return subprocess.Popen(
            [birdnet_script, "listen", "60"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    return None


def main():
    setup_logging()

    config = get_config(mode="batch")

    output_dir = os.path.join(config.skill_dir, "captures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config.detections_dir, exist_ok=True)

    # Load YOLO model
    logger.info("Loading %s...", config.model)
    model = YOLO(config.model)
    logger.info("Model loaded!")

    # Startup banner (user-facing HUD — intentionally print)
    print("=" * 60)
    print("🐦 Bird Watcher v2.0 — YOLO + Moondream Hybrid")
    print(f"   YOLO: {config.model} (bird class, conf>{config.confidence})")
    print(f"   VLM: Moondream Station (species ID on detections)")
    print(f"   Audio: BirdNET (60s cycles)")
    print(f"   Interval: {config.interval}s | Duration: {config.duration}s")
    print("=" * 60)

    start_time = time.time()
    frame_count = 0
    detection_count = 0
    detections = []
    birdnet_proc = run_birdnet_async()

    try:
        while (time.time() - start_time) < config.duration:
            elapsed = int(time.time() - start_time)
            frame_count += 1

            logger.info("[%4ds] Frame %d...", elapsed, frame_count)
            image_path = capture_frame(output_dir)

            if not image_path:
                logger.info("Capture failed")
                time.sleep(config.interval)
                continue

            birds = yolo_detect_birds(model, image_path, config.confidence, config.bird_class_id)

            if not birds:
                logger.info("No birds (YOLO)")
                try:
                    os.remove(image_path)
                except OSError:
                    pass
                time.sleep(config.interval)

                # Cycle BirdNET
                if birdnet_proc and birdnet_proc.poll() is not None:
                    stdout = birdnet_proc.stdout.read().decode() if birdnet_proc.stdout else ""
                    if "detected" in stdout.lower() and "no bird" not in stdout.lower():
                        logger.info("BirdNET: %s", stdout.strip())
                    birdnet_proc = run_birdnet_async()
                continue

            # BIRD DETECTED!
            logger.info("%d bird(s) detected!", len(birds))
            species_labels = []

            for i, bird in enumerate(birds):
                logger.info(
                    "Bird %d: conf=%s, bbox=%s",
                    i + 1, f"{bird['confidence']:.0%}", bird["bbox"],
                )
                species = moondream_identify_batch(
                    image_path, bird["bbox"], config.moondream_url,
                )
                species_labels.append(species)
                logger.info("Moondream ID: %s", species)

                clean_species = species.split(",")[0].split(".")[0].strip()
                log_to_census(clean_species, species)

            annotated_path = annotate_frame(image_path, birds, species_labels)

            detection_count += 1
            detections.append({
                "time": datetime.now().isoformat(),
                "frame": frame_count,
                "elapsed": elapsed,
                "num_birds": len(birds),
                "species": species_labels,
                "annotated_image": annotated_path,
                "original_image": image_path,
            })

            det_filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            det_path = os.path.join(config.detections_dir, det_filename)
            shutil.copy2(annotated_path, det_path)
            logger.info("Saved: %s", det_path)

            # Cycle BirdNET
            if birdnet_proc and birdnet_proc.poll() is not None:
                stdout = birdnet_proc.stdout.read().decode() if birdnet_proc.stdout else ""
                if "detected" in stdout.lower() and "no bird" not in stdout.lower():
                    logger.info("BirdNET: %s", stdout.strip())
                birdnet_proc = run_birdnet_async()

            time.sleep(config.interval)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")

    # Summary (user-facing — intentionally print)
    print("\n" + "=" * 60)
    print("🐦 Bird Watcher v2 Summary")
    print(f"   Frames analyzed: {frame_count}")
    print(f"   Detections: {detection_count}")
    print(f"   Total birds spotted: {sum(d['num_birds'] for d in detections)}")
    print(f"   Duration: {int(time.time() - start_time)}s")
    if detections:
        print(f"\n   Detection Log:")
        for d in detections:
            species_str = ", ".join(d["species"])
            print(f"   • [{d['elapsed']}s] {d['num_birds']} bird(s): {species_str}")
    print("=" * 60)

    # Save session log
    log_path = os.path.join(
        config.detections_dir,
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(log_path, "w") as f:
        json.dump({
            "version": "2.0",
            "pipeline": "YOLO + Moondream + BirdNET",
            "frames": frame_count,
            "detections": detection_count,
            "total_birds": sum(d["num_birds"] for d in detections),
            "duration_seconds": int(time.time() - start_time),
            "results": detections,
        }, f, indent=2)

    return detections


if __name__ == "__main__":
    main()
