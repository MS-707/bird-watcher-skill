#!/usr/bin/env python3
"""
Bird Watcher — Storage module.
Detection frame saving, cleanup, and directory management.
"""

import os
import logging

logger = logging.getLogger("bird-watcher")


def ensure_directories(config):
    """Create required output directories if they don't exist."""
    os.makedirs(config.detections_dir, exist_ok=True)
