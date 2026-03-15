"""Download handler for model files on RunPod serverless workers.

Handles two download sources:
- CivitAI: Uses download_with_aria.py with a model version ID
- Direct URL: Uses aria2c to download from any URL (HuggingFace, etc.)

Files are downloaded to /runpod-volume/ComfyUI/models/<dest>/.
"""

import os
import re
import subprocess
import time

import runpod

MODELS_BASE = "/runpod-volume/ComfyUI/models"
CIVITAI_SCRIPT = "/tools/civitai-downloader/download_with_aria.py"


def _send_progress(job: dict, message: str, percent: float = 0) -> None:
    """Send a progress update to RunPod."""
    try:
        runpod.serverless.progress_update(job, {
            "stage": "download",
            "percent": round(percent, 1),
            "message": message,
        })
    except Exception:
        pass


def _download_civitai(version_id: str, dest_dir: str) -> dict:
    """Download a model from CivitAI using download_with_aria.py.

    Args:
        version_id: CivitAI model version ID.
        dest_dir: Absolute path to destination directory.

    Returns:
        Dict with filename, path, size_mb.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # List files before download to detect the new file
    before = set(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else set()

    result = subprocess.run(
        ["python3", CIVITAI_SCRIPT, "-m", str(version_id), "-o", dest_dir],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CivitAI download failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    # Find newly downloaded file(s)
    after = set(os.listdir(dest_dir)) if os.path.isdir(dest_dir) else set()
    new_files = after - before

    if not new_files:
        raise RuntimeError(
            f"CivitAI download produced no new files. "
            f"stdout: {result.stdout.strip()}"
        )

    # Return info about the first new file (usually there's only one)
    filename = sorted(new_files)[0]
    filepath = os.path.join(dest_dir, filename)
    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)

    return {
        "filename": filename,
        "path": filepath,
        "size_mb": size_mb,
    }


def _parse_aria2c_progress(line: str) -> tuple[float, str] | None:
    """Parse aria2c progress from a summary line.

    aria2c prints lines like:
      [#abc123 1.2GiB/3.5GiB(34%) CN:8 DL:52MiB]
      [#abc123 45MiB/3.5GiB(1%) CN:1 DL:12MiB]

    Returns (percent, speed_str) or None if not a progress line.
    """
    m = re.search(r'\((\d+)%\)', line)
    if not m:
        return None
    pct = int(m.group(1))
    speed = ""
    s = re.search(r'DL:([^\s\]]+)', line)
    if s:
        speed = s.group(1)
    return (pct, speed)


def _download_url(
    url: str,
    dest_dir: str,
    filename: str | None = None,
    job: dict | None = None,
    item_index: int = 0,
    total_items: int = 1,
) -> dict:
    """Download a file from a direct URL using aria2c with progress streaming.

    Args:
        url: Direct download URL.
        dest_dir: Absolute path to destination directory.
        filename: Output filename. If None, derived from URL.
        job: RunPod job dict for progress updates.
        item_index: Current download index (0-based) for progress calculation.
        total_items: Total number of downloads in this batch.

    Returns:
        Dict with filename, path, size_mb.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if not filename:
        filename = url.rstrip("/").rsplit("/", 1)[-1]
        # Strip query params from filename
        if "?" in filename:
            filename = filename.split("?")[0]

    # Stream aria2c output to capture real-time progress
    proc = subprocess.Popen(
        [
            "aria2c", "-d", dest_dir, "-o", filename,
            "--allow-overwrite=true",
            "--summary-interval=3",
            "--console-log-level=notice",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    last_progress_time = 0
    try:
        for line in proc.stdout:
            output_lines.append(line)
            parsed = _parse_aria2c_progress(line)
            if parsed and job:
                dl_pct, speed = parsed
                now = time.time()
                # Throttle progress updates to every 3 seconds
                if now - last_progress_time >= 3:
                    last_progress_time = now
                    # Map download progress into the overall batch progress
                    base_pct = (item_index / total_items) * 100
                    item_pct = (dl_pct / 100) * (100 / total_items)
                    overall_pct = base_pct + item_pct
                    speed_str = f" ({speed}/s)" if speed else ""
                    _send_progress(
                        job,
                        f"Downloading {item_index+1}/{total_items}: "
                        f"{filename} {dl_pct}%{speed_str}",
                        percent=overall_pct,
                    )
    except Exception:
        pass

    proc.wait(timeout=600)

    if proc.returncode != 0:
        full_output = "".join(output_lines).strip()
        raise RuntimeError(
            f"aria2c download failed (exit {proc.returncode}): {full_output}"
        )

    filepath = os.path.join(dest_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(f"Download completed but file not found: {filepath}")

    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)

    return {
        "filename": filename,
        "path": filepath,
        "size_mb": size_mb,
    }


def handle(job: dict) -> dict:
    """Handle a download command job.

    Expected input:
    {
        "command": "download",
        "downloads": [
            {"source": "civitai", "version_id": "12345", "dest": "loras"},
            {"source": "url", "url": "https://...", "dest": "checkpoints", "filename": "model.safetensors"}
        ]
    }

    Returns:
    {
        "ok": true,
        "files": [
            {"filename": "...", "dest": "loras", "path": "...", "size_mb": 123.4}
        ]
    }
    """
    start_time = time.time()
    job_input = job["input"]
    job_id = job.get("id", "unknown")
    downloads = job_input.get("downloads", [])

    if not downloads:
        raise RuntimeError("No downloads specified. Provide a 'downloads' array.")

    # Set CivitAI token if provided in the job payload
    civitai_token = job_input.get("civitai_token", "")
    if civitai_token:
        os.environ["CIVITAI_TOKEN"] = civitai_token

    print(f"[job {job_id[:8]}] Download command: {len(downloads)} file(s)")
    results = []

    for i, dl in enumerate(downloads):
        source = dl.get("source", "")
        dest = dl.get("dest", "checkpoints")
        dest_dir = os.path.join(MODELS_BASE, dest)

        pct = (i / len(downloads)) * 100
        _send_progress(job, f"Downloading {i+1}/{len(downloads)}", percent=pct)

        if source == "civitai":
            version_id = dl.get("version_id")
            if not version_id:
                raise RuntimeError(f"Download {i+1}: 'version_id' required for civitai source")
            print(f"[job {job_id[:8]}] CivitAI download: version {version_id} -> {dest}")
            info = _download_civitai(str(version_id), dest_dir)

        elif source == "url":
            url = dl.get("url")
            if not url:
                raise RuntimeError(f"Download {i+1}: 'url' required for url source")
            filename = dl.get("filename")
            print(f"[job {job_id[:8]}] URL download: {url} -> {dest}")
            info = _download_url(
                url, dest_dir, filename,
                job=job, item_index=i, total_items=len(downloads),
            )

        else:
            raise RuntimeError(f"Download {i+1}: unknown source '{source}'. Use 'civitai' or 'url'.")

        info["dest"] = dest
        results.append(info)
        print(f"[job {job_id[:8]}] Downloaded: {info['filename']} ({info['size_mb']} MB)")

    elapsed = int(time.time() - start_time)
    _send_progress(job, f"Done — {len(results)} file(s) in {elapsed}s", percent=100)
    print(f"[job {job_id[:8]}] Download complete: {len(results)} file(s) in {elapsed}s")

    return {"ok": True, "files": results}
