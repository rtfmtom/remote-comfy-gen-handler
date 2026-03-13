"""Auto-detect and install missing ComfyUI custom nodes from workflow JSON.

Uses ComfyUI-Manager's extension-node-map to resolve class_type → git repo.
Caches the map at /runpod-volume/.node-map-cache.json for fast lookups.

Flow:
    1. Extract all class_type values from workflow
    2. Query ComfyUI /object_info for installed node types
    3. Diff → missing class_types
    4. Look up repos in extension-node-map
    5. git clone + pip install deps
    6. Restart ComfyUI
"""

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
import urllib.error

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/ComfyUI")
CUSTOM_NODES_DIR = os.path.join(COMFYUI_DIR, "custom_nodes")
NODE_MAP_CACHE = "/runpod-volume/.node-map-cache.json"
NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
NODE_MAP_MAX_AGE = 86400  # refresh cache after 24h


def extract_class_types(workflow: dict) -> set[str]:
    """Extract all unique class_type values from an API-format workflow."""
    return {
        node["class_type"]
        for node in workflow.values()
        if isinstance(node, dict) and "class_type" in node
    }


def get_installed_node_types() -> set[str]:
    """Query ComfyUI /object_info for all registered node types."""
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/object_info", timeout=10) as r:
            data = json.loads(r.read())
        return set(data.keys())
    except Exception as e:
        print(f"[node_installer] WARNING: Could not query /object_info: {e}", flush=True)
        return set()


def get_node_map() -> dict:
    """Load extension-node-map, using cache if fresh enough."""
    # Try cache first
    if os.path.exists(NODE_MAP_CACHE):
        age = time.time() - os.path.getmtime(NODE_MAP_CACHE)
        if age < NODE_MAP_MAX_AGE:
            with open(NODE_MAP_CACHE) as f:
                return json.load(f)

    # Fetch from GitHub
    print("[node_installer] Fetching extension-node-map from GitHub...", flush=True)
    try:
        with urllib.request.urlopen(NODE_MAP_URL, timeout=15) as r:
            data = json.loads(r.read())
        os.makedirs(os.path.dirname(NODE_MAP_CACHE), exist_ok=True)
        with open(NODE_MAP_CACHE, "w") as f:
            json.dump(data, f)
        print(f"[node_installer] Cached node map ({len(data)} entries)", flush=True)
        return data
    except Exception as e:
        print(f"[node_installer] WARNING: Could not fetch node map: {e}", flush=True)
        # Fall back to stale cache if available
        if os.path.exists(NODE_MAP_CACHE):
            with open(NODE_MAP_CACHE) as f:
                return json.load(f)
        return {}


def resolve_repos(missing_types: set[str], node_map: dict) -> dict[str, list[str]]:
    """Map missing class_types to git repo URLs.

    Returns {repo_url: [class_type, ...]} for repos that need installing.
    """
    # node_map format: {"repo_url": [["class_type1", "class_type2", ...], {...}]}
    # The first element is a list of class_types provided by that repo.
    repos = {}
    for repo_url, info in node_map.items():
        if not isinstance(info, list) or len(info) == 0:
            continue
        provided_types = info[0] if isinstance(info[0], list) else []
        matched = missing_types & set(provided_types)
        if matched:
            repos[repo_url] = list(matched)
    return repos


def install_repo(repo_url: str, force_deps: bool = False) -> bool:
    """Clone a custom node repo and install its dependencies.

    Args:
        repo_url: Git URL of the custom node repo.
        force_deps: If True, reinstall deps even if repo dir already exists.
                    Used when the repo exists but its nodes failed to import.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    target = os.path.join(CUSTOM_NODES_DIR, repo_name)

    if os.path.exists(target) and not force_deps:
        print(f"[node_installer] {repo_name} already exists, skipping clone", flush=True)
        return False

    if os.path.exists(target) and force_deps:
        print(f"[node_installer] {repo_name} exists but nodes not loaded — reinstalling deps", flush=True)
    else:
        print(f"[node_installer] Installing {repo_name} from {repo_url}", flush=True)
        # Clone
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"[node_installer] ERROR cloning {repo_name}: {result.stderr[:500]}", flush=True)
            return False

    # Install requirements.txt if present
    req_file = os.path.join(target, "requirements.txt")
    if os.path.exists(req_file):
        print(f"[node_installer] Installing deps for {repo_name}...", flush=True)
        subprocess.run(
            ["pip", "install", "-q", "-r", req_file],
            capture_output=True, text=True, timeout=300,
        )

    # Run install.py if present
    install_script = os.path.join(target, "install.py")
    if os.path.exists(install_script):
        print(f"[node_installer] Running install.py for {repo_name}...", flush=True)
        subprocess.run(
            ["python3", install_script],
            capture_output=True, text=True, timeout=120,
            cwd=target,
        )

    print(f"[node_installer] {repo_name} installed successfully", flush=True)
    return True


def restart_comfyui() -> bool:
    """Kill ComfyUI and restart it fresh. Wait until ready."""
    print("[node_installer] Restarting ComfyUI...", flush=True)

    # Find and kill ComfyUI process
    result = subprocess.run(
        ["pkill", "-f", "python3 main.py"],
        capture_output=True, text=True,
    )
    time.sleep(2)

    # Start ComfyUI in background
    comfy_port = os.environ.get("COMFYUI_PORT", "8188")
    cmd = [
        "python3", "main.py",
        "--listen", "0.0.0.0",
        "--port", comfy_port,
        "--disable-auto-launch",
        "--disable-metadata",
    ]
    extra_paths = os.path.join(COMFYUI_DIR, "extra_model_paths.yaml")
    if os.path.exists(extra_paths):
        cmd += ["--extra-model-paths-config", extra_paths]
    subprocess.Popen(
        cmd,
        cwd=COMFYUI_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for ready
    max_wait = 120
    waited = 0
    while waited < max_wait:
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3) as r:
                r.read()
            print(f"[node_installer] ComfyUI ready after {waited}s", flush=True)
            return True
        except Exception:
            time.sleep(2)
            waited += 2

    print(f"[node_installer] ERROR: ComfyUI failed to restart within {max_wait}s", flush=True)
    return False


def parse_missing_node_from_error(error_msg: str) -> str | None:
    """Extract missing node class_type from ComfyUI error message."""
    # Pattern: "Cannot execute because node X does not exist."
    m = re.search(r"node (\S+) does not exist", error_msg)
    return m.group(1) if m else None


def ensure_nodes(workflow: dict, max_retries: int = 3) -> list[str]:
    """Check workflow for missing nodes and install them.

    Returns list of newly installed repo names.
    Can be called before queue_prompt as a pre-check,
    or after a failure to handle warm-worker missing nodes.
    """
    installed_types = get_installed_node_types()
    required_types = extract_class_types(workflow)
    missing = required_types - installed_types

    if not missing:
        return []

    print(f"[node_installer] Missing node types: {missing}", flush=True)

    node_map = get_node_map()
    repos = resolve_repos(missing, node_map)

    if not repos:
        unresolved = missing - {t for types in repos.values() for t in types}
        if unresolved:
            print(f"[node_installer] WARNING: Could not resolve repos for: {unresolved}", flush=True)
        return []

    # Install all missing repos (force deps if dir exists but nodes aren't loaded)
    installed_repos = []
    for repo_url, types in repos.items():
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        target = os.path.join(CUSTOM_NODES_DIR, repo_name)
        dir_exists = os.path.exists(target)
        print(f"[node_installer] {repo_url} provides: {types}", flush=True)
        if install_repo(repo_url, force_deps=dir_exists):
            installed_repos.append(repo_name)

    # Restart ComfyUI to pick up new nodes
    if installed_repos:
        if not restart_comfyui():
            raise RuntimeError("Failed to restart ComfyUI after installing nodes")

    return installed_repos
