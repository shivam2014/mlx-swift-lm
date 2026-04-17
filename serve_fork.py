#!/usr/bin/env python3
"""
MLXServer Launcher — fork-specific wrapper.
Only launches MLXServer (simplified from upstream serve.py option 6).

USAGE:
  python3 scripts/serve_fork.py

CONFIGURATION:
  Set FORK_MODEL_PATH environment variable to override model location.
  Default: ~/.cache/huggingface/hub/thetom-ai/Qwen3.6-35B-A3B-ConfigI-MLX

EXAMPLE:
  export FORK_MODEL_PATH="/path/to/your/model"
  python3 scripts/serve_fork.py
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request

# =============================================================================
# CONFIG (override via environment)
# =============================================================================

MODEL = os.environ.get(
    "FORK_MODEL_PATH",
    os.path.expanduser(
        "/Users/shivam94/.cache/huggingface/hub/thetom-ai/Qwen3.6-35B-A3B-ConfigI-MLX"
    )
)
PORT = 8000
HOST = "127.0.0.1"

MLX_ENV = {}

MLXSERVER = os.path.expanduser(
    "~/mlx-env/mlx-swift-lm/.build/release/MLXServer"
)

# =============================================================================

def wait_server(host, port, timeout=300):
    base = f"http://{host}:{port}"
    for _ in range(timeout):
        for ep in ["/health", "/v1/models"]:
            try:
                urllib.request.urlopen(f"{base}{ep}", timeout=2)
                return True
            except Exception:
                pass
        time.sleep(1)
    return False


def kill_port(port):
    os.system(f"lsof -ti tcp:{port} | xargs kill -9 2>/dev/null")
    time.sleep(1)


def main():
    print("=" * 60)
    print("  MLXServer Launcher (fork)")
    print(f"  Model : {os.path.basename(MODEL)}")
    print(f"  Path  : {MODEL}")
    print(f"  Host  : {HOST}   Port: {PORT}")
    print("=" * 60)
    print()

    # Build command (matches _mlxserver_cmd from upstream serve.py)
    cmd = [
        MLXSERVER,
        "--model", MODEL,
        "--port", str(PORT),
        "--slots", "4",
        "--kv", "turbo4v2",
    ]

    print(f"  CMD: {' '.join(cmd)}")
    print()

    kill_port(PORT)
    time.sleep(1)

    env = os.environ.copy()
    env.update(MLX_ENV)

    proc = subprocess.Popen(cmd, env=env)

    def _shutdown(sig, frame):
        print(f"\n  Stopping MLXServer…")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        kill_port(PORT)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"  Waiting for server on {HOST}:{PORT} (up to 5 min)…")
    if wait_server(HOST, PORT, timeout=300):
        print(f"\n  Server ready → http://{HOST}:{PORT}/v1")
        print(f"  Endpoint  : http://{HOST}:{PORT}/v1/chat/completions")
        print(f"  Model ID  : {MODEL}")
        print(f"  Kill      : Ctrl-C")
        print()
    else:
        print("  [warn] server did not respond within 5 min — check logs above")

    try:
        proc.wait()
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
