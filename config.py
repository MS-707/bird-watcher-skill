#!/usr/bin/env python3
"""
Bird Watcher — Configuration module.
Handles argparse CLI, environment variable overrides, and shared constants.
"""

import argparse
import collections
import logging
import os
import secrets
import threading

logger = logging.getLogger("bird-watcher")

# COCO class ID for 'bird'
BIRD_CLASS_ID = 14

# Paths — use relative/expanduser, never hardcoded usernames
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTIONS_DIR = os.path.join(SKILL_DIR, "detections")
CENSUS_SCRIPT = os.path.expanduser("~/.openclaw/skills/wildlife-census/census.sh")


def _env_int(name, default):
    """Read an integer from environment, falling back to default."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            logger.warning("Invalid integer for %s=%r, using default %d", name, val, default)
    return default


def _env_float(name, default):
    """Read a float from environment, falling back to default."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            logger.warning("Invalid float for %s=%r, using default %s", name, val, default)
    return default


def _env_str(name, default):
    """Read a string from environment, falling back to default."""
    return os.environ.get(name, default)


def build_stream_parser():
    """Build argparse parser for the live stream mode."""
    parser = argparse.ArgumentParser(
        description="Bird Watcher Live Stream — decoupled camera + YOLO architecture"
    )
    parser.add_argument(
        "--port", type=int,
        default=_env_int("BIRDWATCH_PORT", 8888),
        help="HTTP port for the MJPEG stream server (default: 8888, env: BIRDWATCH_PORT)"
    )
    parser.add_argument(
        "--model", type=str,
        default=_env_str("BIRDWATCH_MODEL", "yolo11s.pt"),
        help="YOLO model file name (default: yolo11s.pt, env: BIRDWATCH_MODEL)"
    )
    parser.add_argument(
        "--confidence", type=float,
        default=_env_float("BIRDWATCH_CONFIDENCE", 0.15),
        help="YOLO confidence threshold (default: 0.15, env: BIRDWATCH_CONFIDENCE)"
    )
    parser.add_argument(
        "--persist", type=int,
        default=_env_int("BIRDWATCH_PERSIST", 3),
        help="Seconds to keep bounding boxes on screen after detection (default: 3, env: BIRDWATCH_PERSIST)"
    )
    parser.add_argument(
        "--no-save", action="store_true", default=False,
        help="Disable saving detection frames to disk"
    )
    return parser


def build_batch_parser():
    """Build argparse parser for the batch detection mode."""
    parser = argparse.ArgumentParser(
        description="Bird Watcher Batch — YOLO + Moondream hybrid detection"
    )
    parser.add_argument(
        "--duration", type=int,
        default=_env_int("BIRDWATCH_DURATION", 1800),
        help="Total run duration in seconds (default: 1800, env: BIRDWATCH_DURATION)"
    )
    parser.add_argument(
        "--interval", type=int,
        default=_env_int("BIRDWATCH_INTERVAL", 8),
        help="Seconds between frame captures (default: 8, env: BIRDWATCH_INTERVAL)"
    )
    parser.add_argument(
        "--model", type=str,
        default=_env_str("BIRDWATCH_MODEL", "yolo11s.pt"),
        help="YOLO model file name (default: yolo11s.pt, env: BIRDWATCH_MODEL)"
    )
    parser.add_argument(
        "--confidence", type=float,
        default=_env_float("BIRDWATCH_CONFIDENCE", 0.15),
        help="YOLO confidence threshold (default: 0.15, env: BIRDWATCH_CONFIDENCE)"
    )
    parser.add_argument(
        "--no-save", action="store_true", default=False,
        help="Disable saving detection frames to disk"
    )
    return parser


def get_config(mode="stream"):
    """
    Parse CLI args + env vars and return a config namespace.

    Parameters
    ----------
    mode : str
        'stream' for live stream, 'batch' for batch detection.

    Returns
    -------
    argparse.Namespace with all configuration values.
    """
    if mode == "batch":
        parser = build_batch_parser()
    else:
        parser = build_stream_parser()

    args = parser.parse_args()

    # Attach additional env-only settings as attributes
    args.stream_token = _env_str("BIRDWATCH_TOKEN", secrets.token_urlsafe(16))
    args.max_detection_files = _env_int("BIRDWATCH_MAX_FILES", 500)
    args.max_concurrent_viewers = _env_int("BIRDWATCH_MAX_VIEWERS", 5)
    args.moondream_url = _env_str("MOONDREAM_URL", "http://localhost:2020")
    args.min_bird_size = _env_int("BIRDWATCH_MIN_BIRD_SIZE", 50)

    # Shared path constants
    args.skill_dir = SKILL_DIR
    args.detections_dir = DETECTIONS_DIR
    args.census_script = CENSUS_SCRIPT
    args.bird_class_id = BIRD_CLASS_ID

    return args


def setup_logging(level=logging.INFO):
    """Configure structured logging for the bird-watcher package."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Thread-safe stats lock — acquire before reading or writing stats dict values.
stats_lock = threading.Lock()


def make_stats():
    """Create a fresh stats dictionary with thread-safe detection_log."""
    return {
        "camera_fps": 0,
        "yolo_fps": 0,
        "bird_count": 0,
        "total_detections": 0,
        "last_species": "Bird",
        "detection_log": collections.deque(maxlen=100),
    }
