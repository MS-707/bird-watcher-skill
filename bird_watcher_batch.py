#!/usr/bin/env python3
"""
Bird Watcher v2 — YOLO + Moondream Hybrid Detection
- YOLO for fast bird detection with bounding boxes
- Moondream VLM for species identification on detected birds
- BirdNET for audio species detection
- Sends annotated photos to Telegram on detection

Usage: python3 bird_watcher_v2.py [--duration 1800] [--interval 5]
"""

import subprocess
import requests
import json
import time
import os
import sys
import base64
import shutil
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Config
MOONDREAM_URL = "http://localhost:2020"
CAPTURE_INTERVAL = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 8
DURATION = int(sys.argv[sys.argv.index("--duration") + 1]) if "--duration" in sys.argv else 1800
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SKILL_DIR, "captures")
DETECTIONS_DIR = os.path.join(SKILL_DIR, "detections")
CENSUS_SCRIPT = os.path.expanduser("~/.openclaw/skills/wildlife-census/census.sh")
SNAP_DIR = "/tmp/victor_vision"

# YOLO bird class ID in COCO = 14
BIRD_CLASS_ID = 14
CONFIDENCE_THRESHOLD = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Load YOLO model (nano for speed on M1)
print("Loading YOLOv11-nano model...")
model = YOLO("yolo11n.pt")
print("Model loaded!")


def capture_frame():
    """Capture via snap watcher (has camera permissions)."""
    os.makedirs(SNAP_DIR, exist_ok=True)
    ready_file = os.path.join(SNAP_DIR, "ready")
    try:
        os.remove(ready_file)
    except:
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
                dst_path = f"{OUTPUT_DIR}/frame_{timestamp}.jpg"
                shutil.copy2(src_path, dst_path)
                return dst_path
            return None
        time.sleep(0.5)
    return None


def yolo_detect_birds(image_path):
    """Run YOLO on frame, return bird detections."""
    results = model(image_path, verbose=False, conf=CONFIDENCE_THRESHOLD)
    birds = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == BIRD_CLASS_ID:
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

        # Green bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)

        # Label
        label = f"Bird {conf:.0%}"
        if species_labels and i < len(species_labels):
            label = f"{species_labels[i]} {conf:.0%}"

        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 6, y1), (0, 255, 100), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Timestamp overlay
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, f"Bird Watcher v2 | {ts}", (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    # Bird count
    cv2.putText(img, f"Birds: {len(birds)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

    annotated_path = image_path.replace("/frame_", "/annotated_")
    cv2.imwrite(annotated_path, img)
    return annotated_path


def moondream_identify(image_path, bbox):
    """Crop bird region and ask Moondream for species ID."""
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown bird"

    x1, y1, x2, y2 = bbox
    # Expand crop region by 20% for context
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * 0.2)
    pad_y = int((y2 - y1) * 0.2)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    crop = img[cy1:cy2, cx1:cx2]

    # Encode crop
    _, buffer = cv2.imencode(".jpg", crop)
    img_b64 = base64.b64encode(buffer).decode()

    try:
        resp = requests.post(
            f"{MOONDREAM_URL}/caption",
            json={
                "image": img_b64,
                "prompt": "What species of bird is this? Identify the species name. Be specific and concise — just the species name and one brief detail about its appearance."
            },
            timeout=15
        )
        if resp.ok:
            data = resp.json()
            caption = data.get("caption", data.get("result", "Unknown bird"))
            return caption.strip()
    except Exception as e:
        print(f"   Moondream error: {e}")
    return "Unknown bird"


def log_to_census(species, caption):
    """Log detection to wildlife census."""
    try:
        subprocess.run(
            [CENSUS_SCRIPT, "log", species, "1", f"YOLO+Moondream: {caption[:100]}"],
            capture_output=True, timeout=10
        )
    except:
        pass


def run_birdnet_async():
    """Start BirdNET listening in background."""
    birdnet_script = os.path.expanduser("~/.openclaw/skills/birdnet-audio/birdnet.sh")
    if os.path.exists(birdnet_script):
        print("🎤 BirdNET recording (60s)...")
        return subprocess.Popen(
            [birdnet_script, "listen", "60"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    return None


def main():
    print("=" * 60)
    print("🐦 Bird Watcher v2.0 — YOLO + Moondream Hybrid")
    print(f"   YOLO: YOLOv11-nano (bird class, conf>{CONFIDENCE_THRESHOLD})")
    print(f"   VLM: Moondream Station (species ID on detections)")
    print(f"   Audio: BirdNET (60s cycles)")
    print(f"   Interval: {CAPTURE_INTERVAL}s | Duration: {DURATION}s")
    print("=" * 60)

    start_time = time.time()
    frame_count = 0
    detection_count = 0
    detections = []
    birdnet_proc = run_birdnet_async()

    try:
        while (time.time() - start_time) < DURATION:
            elapsed = int(time.time() - start_time)
            frame_count += 1

            print(f"[{elapsed:4d}s] Frame {frame_count}... ", end="", flush=True)
            image_path = capture_frame()

            if not image_path:
                print("capture failed")
                time.sleep(CAPTURE_INTERVAL)
                continue

            # YOLO detection (fast)
            birds = yolo_detect_birds(image_path)

            if not birds:
                print("no birds (YOLO)")
                try:
                    os.remove(image_path)
                except:
                    pass
                time.sleep(CAPTURE_INTERVAL)

                # Cycle BirdNET
                if birdnet_proc and birdnet_proc.poll() is not None:
                    stdout = birdnet_proc.stdout.read().decode() if birdnet_proc.stdout else ""
                    if "detected" in stdout.lower() and "no bird" not in stdout.lower():
                        print(f"   🎤 BirdNET: {stdout.strip()}")
                    birdnet_proc = run_birdnet_async()
                continue

            # BIRD DETECTED! Run Moondream on each detection
            print(f"🐦 {len(birds)} bird(s) detected!")
            species_labels = []

            for i, bird in enumerate(birds):
                print(f"   Bird {i+1}: conf={bird['confidence']:.0%}, bbox={bird['bbox']}")
                species = moondream_identify(image_path, bird["bbox"])
                species_labels.append(species)
                print(f"   → Moondream ID: {species}")

                # Extract clean species name for census
                clean_species = species.split(",")[0].split(".")[0].strip()
                log_to_census(clean_species, species)

            # Annotate frame with boxes and labels
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

            # Copy annotated image to detections folder
            det_filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            det_path = os.path.join(DETECTIONS_DIR, det_filename)
            shutil.copy2(annotated_path, det_path)
            print(f"   📸 Saved: {det_path}")

            # Cycle BirdNET
            if birdnet_proc and birdnet_proc.poll() is not None:
                stdout = birdnet_proc.stdout.read().decode() if birdnet_proc.stdout else ""
                if "detected" in stdout.lower() and "no bird" not in stdout.lower():
                    print(f"   🎤 BirdNET: {stdout.strip()}")
                birdnet_proc = run_birdnet_async()

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    # Summary
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
    log_path = f"{DETECTIONS_DIR}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump({
            "version": "2.0",
            "pipeline": "YOLO + Moondream + BirdNET",
            "frames": frame_count,
            "detections": detection_count,
            "total_birds": sum(d["num_birds"] for d in detections),
            "duration_seconds": int(time.time() - start_time),
            "results": detections
        }, f, indent=2)

    return detections


if __name__ == "__main__":
    main()
