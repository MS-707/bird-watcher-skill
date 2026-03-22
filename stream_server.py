#!/usr/bin/env python3
"""
Bird Watcher — Flask stream server module.
MJPEG streaming server with token-based authentication and viewer limits.
"""

import time
import logging
import threading

from flask import Flask, Response, render_template_string, request, abort

logger = logging.getLogger("bird-watcher")

app = Flask(__name__)

# Module-level state — set by init_server() before app.run()
_config = None
_shared_state = None
_active_viewers = 0
_viewers_lock = threading.Lock()

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


def init_server(config, shared_state):
    """
    Initialize the Flask server with configuration and shared state.

    Must be called before app.run().
    """
    global _config, _shared_state
    _config = config
    _shared_state = shared_state


def _generate_mjpeg():
    """Yield MJPEG frames from the shared state."""
    while True:
        with _shared_state["stream_lock"]:
            frame = _shared_state["stream_jpeg"]
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    token = request.args.get('token', '')
    if token != _config.stream_token:
        return (
            '<html><body style="background:#000;color:#fff;font-family:sans-serif;'
            'display:flex;align-items:center;justify-content:center;height:100vh">'
            '<div><h1>🐦 Bird Watcher</h1>'
            '<p>Access token required. Add ?token=YOUR_TOKEN to the URL.</p>'
            '</div></body></html>'
        ), 403
    page = HTML_PAGE.replace('/feed', f'/feed?token={_config.stream_token}')
    return render_template_string(page)


@app.route('/feed')
def video_feed():
    global _active_viewers
    token = request.args.get('token', '')
    if token != _config.stream_token:
        abort(403)
    with _viewers_lock:
        if _active_viewers >= _config.max_concurrent_viewers:
            abort(503)
        _active_viewers += 1
    try:
        return Response(
            _generate_mjpeg(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    finally:
        with _viewers_lock:
            _active_viewers = max(0, _active_viewers - 1)
