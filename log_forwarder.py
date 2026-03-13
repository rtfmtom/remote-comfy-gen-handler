#!/usr/bin/env python3
"""Reads stdin line-by-line, echoes to stdout, and batches+forwards to a remote log receiver.

Used on RunPod workers to forward logs to a local machine via Cloudflare tunnel.

Usage (in start.sh):
    exec torchrun ... 2>&1 | python log_forwarder.py

Environment variables:
    LOG_RECEIVER_URL   - Full URL of log receiver (e.g. https://xxx.trycloudflare.com)
    LOG_RECEIVER_TOKEN - Bearer token for auth
    RUNPOD_ENDPOINT_ID - RunPod endpoint ID (for log grouping)
    RUNPOD_POD_ID      - RunPod pod/worker ID (for log grouping)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error

URL = os.environ.get("LOG_RECEIVER_URL", "").rstrip("/")
TOKEN = os.environ.get("LOG_RECEIVER_TOKEN", "")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", os.environ.get("RUNPOD_DC_ID", "unknown"))
WORKER_ID = os.environ.get("RUNPOD_POD_ID", os.environ.get("HOSTNAME", "unknown"))

BATCH_SIZE = 50
FLUSH_INTERVAL = 2.0  # seconds

_buffer: list[str] = []
_lock = threading.Lock()
_stop = threading.Event()


def _flush():
    """Send buffered lines to the receiver."""
    with _lock:
        if not _buffer:
            return
        lines = _buffer.copy()
        _buffer.clear()

    payload = json.dumps({
        "endpoint_id": ENDPOINT_ID,
        "worker_id": WORKER_ID,
        "lines": lines,
    }).encode()

    req = urllib.request.Request(
        URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "User-Agent": "runpod-log-forwarder/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    except Exception as e:
        # Log to stderr so it shows in RunPod logs but doesn't crash the worker
        print(f"[log-forwarder] flush error: {e}", file=sys.stderr, flush=True)


def _flush_loop():
    """Background thread that flushes on interval."""
    while not _stop.is_set():
        _stop.wait(FLUSH_INTERVAL)
        _flush()


def main():
    if not URL:
        # No receiver configured — just pass stdin through to stdout
        for line in sys.stdin:
            sys.stdout.write(line)
            sys.stdout.flush()
        return

    print(f"[log-forwarder] started: url={URL} endpoint={ENDPOINT_ID} worker={WORKER_ID}", file=sys.stderr, flush=True)

    # Start background flush thread
    t = threading.Thread(target=_flush_loop, daemon=True)
    t.start()

    try:
        for line in sys.stdin:
            # Always echo to stdout (RunPod captures stdout for its own logs)
            sys.stdout.write(line)
            sys.stdout.flush()

            stripped = line.rstrip("\n")
            with _lock:
                _buffer.append(stripped)
                should_flush = len(_buffer) >= BATCH_SIZE
            if should_flush:
                _flush()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        _flush()  # Final flush


if __name__ == "__main__":
    main()
