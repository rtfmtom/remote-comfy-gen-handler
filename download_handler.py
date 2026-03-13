"""Download handler for model files on RunPod serverless workers.

Handles two download sources:
- CivitAI: Uses download_with_aria.py with a model version ID
- Direct URL: Uses aria2c to download from any URL (HuggingFace, etc.)

Files are downloaded to /runpod-volume/ComfyUI/models/<dest>/.
"""

import os
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
        ["python3", CIVITAI_SCRIPT, "-v", str(version_id), "-o", dest_dir],
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


def _download_url(url: str, dest_dir: str, filename: str | None = None) -> dict:
    """Download a file from a direct URL using aria2c.

    Args:
        url: Direct download URL.
        dest_dir: Absolute path to destination directory.
        filename: Output filename. If None, derived from URL.

    Returns:
        Dict with filename, path, size_mb.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if not filename:
        filename = url.rstrip("/").rsplit("/", 1)[-1]
        # Strip query params from filename
        if "?" in filename:
            filename = filename.split("?")[0]

    result = subprocess.run(
        ["aria2c", "-d", dest_dir, "-o", filename, "--allow-overwrite=true", url],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"aria2c download failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
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
            info = _download_url(url, dest_dir, filename)

        else:
            raise RuntimeError(f"Download {i+1}: unknown source '{source}'. Use 'civitai' or 'url'.")

        info["dest"] = dest
        results.append(info)
        print(f"[job {job_id[:8]}] Downloaded: {info['filename']} ({info['size_mb']} MB)")

    elapsed = int(time.time() - start_time)
    _send_progress(job, f"Done — {len(results)} file(s) in {elapsed}s", percent=100)
    print(f"[job {job_id[:8]}] Download complete: {len(results)} file(s) in {elapsed}s")

    return {"ok": True, "files": results}
