#!/usr/bin/env python3
"""
Bird Watcher — Moondream species identification module.
Handles VLM-based bird species identification and wildlife census logging.
"""

import os
import base64
import logging
import subprocess

import cv2
import requests

logger = logging.getLogger("bird-watcher")


def verify_moondream(moondream_url):
    """
    Check if Moondream Station is running and healthy.

    Returns True if the server responds correctly.
    """
    try:
        resp = requests.get(f"{moondream_url}/health", timeout=3)
        if resp.ok and resp.json().get("server") == "moondream-station":
            return True
    except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
        logger.debug("Moondream health check failed: %s", exc)
    return False


def moondream_identify(img, bbox, config, stats, moondream_available):
    """
    Species ID via Moondream VLM.

    Parameters
    ----------
    img : numpy.ndarray
        Full camera frame.
    bbox : tuple
        (x1, y1, x2, y2) bounding box of the detected bird.
    config : argparse.Namespace
        Configuration namespace.
    stats : dict
        Shared stats dictionary (mutated in-place).
    moondream_available : bool
        Whether Moondream is reachable.
    """
    if not moondream_available:
        return

    from datetime import datetime

    try:
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        pad = 0.25
        crop = img[
            max(0, int(y1 - (y2 - y1) * pad)):min(h, int(y2 + (y2 - y1) * pad)),
            max(0, int(x1 - (x2 - x1) * pad)):min(w, int(x2 + (x2 - x1) * pad))
        ]
        if crop.size == 0:
            return

        _, buffer = cv2.imencode(".jpg", crop)
        img_b64 = base64.b64encode(buffer).decode()

        resp = requests.post(
            f"{config.moondream_url}/caption",
            json={
                "image": img_b64,
                "prompt": "Identify this bird species. Just the name and one brief detail.",
            },
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            species = data.get("caption", data.get("result", "")).strip()

            bad_answers = [
                "unknown", "not sure", "can't tell", "i can't", "cannot",
                "unclear", "hard to", "difficult to", "image", "photo", "picture",
            ]
            if species and not any(bad in species.lower() for bad in bad_answers):
                stats["last_species"] = species

            stats["detection_log"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "species": species,
            })
            if len(stats["detection_log"]) > 100:
                stats["detection_log"].pop(0)

            # Log to wildlife census
            clean = species.split(",")[0].split(".")[0].strip()
            _log_to_census(clean, species, config.census_script)

    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.warning("Moondream request failed: %s", exc)
    except (cv2.error, ValueError) as exc:
        logger.warning("Image processing error during species ID: %s", exc)


def _log_to_census(species_name, full_caption, census_script):
    """Log a detection to the wildlife census script if it exists."""
    if not os.path.exists(census_script):
        return
    try:
        subprocess.run(
            [census_script, "log", species_name, "1", f"BirdWatcher: {full_caption[:80]}"],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Census logging failed: %s", exc)
