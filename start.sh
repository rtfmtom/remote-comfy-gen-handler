#!/bin/bash
set -euo pipefail

COMFYUI_DIR="/ComfyUI"
COMFYUI_PORT=8188
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFY_LOG="/tmp/comfyui_startup.log"

echo "[start] Booting ComfyUI from $COMFYUI_DIR (baked in image)..."

# Verify ComfyUI exists
if [ ! -f "$COMFYUI_DIR/main.py" ]; then
    echo "[start] ERROR: ComfyUI not found at $COMFYUI_DIR"
    exit 1
fi

# --- Force ComfyUI-Manager offline mode (skip remote fetches, security scan, alembic) ---
MANAGER_CONFIG_DIR="$COMFYUI_DIR/user/__manager"
mkdir -p "$MANAGER_CONFIG_DIR"
if [ ! -f "$MANAGER_CONFIG_DIR/config.ini" ] || ! grep -q "network_mode = offline" "$MANAGER_CONFIG_DIR/config.ini" 2>/dev/null; then
    cat > "$MANAGER_CONFIG_DIR/config.ini" <<'MGREOF'
[default]
network_mode = offline
security_level = normal
MGREOF
    echo "[start] Set ComfyUI-Manager to offline mode"
fi

# Build extra model paths flag — models live on the network volume, not in the image
EXTRA_PATHS_FLAG=""
if [ -f "$COMFYUI_DIR/extra_model_paths.yaml" ]; then
    # Patch in any missing model types without rebuilding the Docker image
    if ! grep -q "detection:" "$COMFYUI_DIR/extra_model_paths.yaml"; then
        sed -i '/^    vae:/i\    detection: detection' "$COMFYUI_DIR/extra_model_paths.yaml"
        echo "[start] Patched extra_model_paths.yaml with detection"
    fi
    EXTRA_PATHS_FLAG="--extra-model-paths-config $COMFYUI_DIR/extra_model_paths.yaml"
    echo "[start] Using extra_model_paths.yaml for network volume models"
fi

# --- Start ComfyUI, tee output to log file for IMPORT FAILED detection ---
cd "$COMFYUI_DIR"
python3 main.py \
    --listen 0.0.0.0 \
    --port $COMFYUI_PORT \
    --disable-auto-launch \
    --disable-metadata \
    $EXTRA_PATHS_FLAG \
    > >(tee "$COMFY_LOG") 2>&1 &

COMFYUI_PID=$!
echo "[start] ComfyUI starting (PID: $COMFYUI_PID)..."

# Wait for ComfyUI to be ready
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; then
        echo "[start] ComfyUI ready after ${WAITED}s"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[start] ERROR: ComfyUI failed to start within ${MAX_WAIT}s"
    kill $COMFYUI_PID 2>/dev/null || true
    exit 1
fi

# --- Fix broken custom nodes (IMPORT FAILED) ---
# Parse ComfyUI startup log for nodes that failed to import,
# reinstall their deps, and restart ComfyUI if any were fixed.
BROKEN_NODES=$(grep -o 'IMPORT FAILED.*custom_nodes/[^"]*' "$COMFY_LOG" 2>/dev/null \
    | sed 's|.*/custom_nodes/||' | sort -u || true)

if [ -n "$BROKEN_NODES" ]; then
    echo "[start] Found broken custom nodes:"
    NEEDS_RESTART=false
    CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"

    while IFS= read -r node_name; do
        req_file="$CUSTOM_NODES_DIR/$node_name/requirements.txt"
        if [ -f "$req_file" ]; then
            echo "[start]   -> $node_name: reinstalling deps..."
            pip install -q -r "$req_file" 2>/dev/null && NEEDS_RESTART=true \
                || echo "[start]   WARNING: deps install failed for $node_name"
        else
            echo "[start]   -> $node_name: no requirements.txt, skipping"
        fi
    done <<< "$BROKEN_NODES"

    if $NEEDS_RESTART; then
        echo "[start] Restarting ComfyUI to reload fixed nodes..."
        kill $COMFYUI_PID 2>/dev/null || true
        sleep 2

        cd "$COMFYUI_DIR"
        python3 main.py \
            --listen 0.0.0.0 \
            --port $COMFYUI_PORT \
            --disable-auto-launch \
            --disable-metadata \
            &
        COMFYUI_PID=$!

        WAITED=0
        while [ $WAITED -lt $MAX_WAIT ]; do
            if curl -s "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; then
                echo "[start] ComfyUI restarted after ${WAITED}s"
                break
            fi
            sleep 2
            WAITED=$((WAITED + 2))
        done

        # Invalidate deps stamp so start_script.sh reinstalls next time
        rm -f /runpod-volume/.custom-node-deps-stamp
    fi
else
    echo "[start] All custom nodes loaded OK"
fi

# Start the RunPod worker handler
echo "[start] Starting RunPod handler..."
exec python3 "$RUNTIME_DIR/worker.py"
