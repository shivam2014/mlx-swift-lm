# mlx-swift-lm (fork)

**Fork of:** https://github.com/ekryski/mlx-swift-lm  
**Your fork:** https://github.com/shivam2014/mlx-swift-lm  
**Branch:** `ek/tom-eric-moe-tuning`  
**Purpose:** Production-ready MLX inference server with custom fixes and experimental features for Qwen3.6 and beyond.

## Why this fork exists

Upstream `ekryski/mlx-swift-lm` is an excellent Swift-based LLM server with Metal acceleration, but certain models require adjustments:

- **Qwen3.6 Config-I models**: Per-tensor quantization lookup was broken → LM head shape mismatches. Fixed in `4537a49`.
- **Thinking mode (Qwen3)**: Passing `enable_thinking: true` alongside `TAG_think` in chat template causes HTTP 400. Fixed in `a439af0` with proper detection.
- **KV cache control**: Added `--kv` flag to select compression schemes (e.g., `turbo4v2`). Fixed in `fde5fd4`.

This fork stabilizes Qwen3.6-35B-A3B Config-I and provides a clean base for further R&D (speculative prefill, cross-family token importance, etc.).

## Getting Started

### Prerequisites

- macOS 15+ (Apple Silicon)
- Xcode command line tools: `xcode-select --install`
- Swift 6.3+
- 64GB RAM (for 35B models with context)
- Git

### 1. Clone Your Fork

```bash
git clone https://github.com/shivam2014/mlx-swift-lm.git
cd mlx-swift-lm
git checkout ek/tom-eric-moe-tuning
```

### 2. Build MLXServer

```bash
./update_mlxserver_fork.sh
```

This script:
- Switches to `ek/tom-eric-moe-tuning` branch
- Pulls latest from **your fork** (`fork` remote)
- Resolves Swift packages
- Compiles Metal shaders (`make metal`)
- Builds `MLXServer` binary in `.build/release/`

Optional: `./update_mlxserver_fork.sh --fresh` to wipe and rebuild.

### 3. Prepare Model

Download an MLX-format model. Two options:

#### Option A: Qwen3.6-35B-A3B Config-I (this fork's test model)

35B parameter Mixture-of-Experts (MoE) model with mixed per-tensor quantization.

- HuggingFace: https://huggingface.co/thetom-ai/Qwen3.6-35B-A3B-ConfigI-MLX
- Download with `huggingface-cli`:

```bash
# New Hugging Face CLI (hf). Install: brew install hf
hf download thetom-ai/Qwen3.6-35B-A3B-ConfigI-MLX --local-dir ~/.cache/huggingface/hub/thetom-ai/Qwen3.6-35B-A3B-ConfigI-MLX
```

#### Option B: Qwen3.5-0.8B (lightweight test)

For quick checks and speculative prefill draft:

- https://huggingface.co/mlx-community/Qwen3.5-0.8B-MLX-4bit-fp16

### 4. Launch Server

```bash
# Set model path if not using default
export FORK_MODEL_PATH="$HOME/.cache/huggingface/hub/thetom-ai/Qwen3.6-35B-A3B-ConfigI-MLX"

# Start server
python3 scripts/serve_fork.py
```

This launches MLXServer on `http://127.0.0.1:8000/v1` with flags:
```
--slots 4
--kv turbo4v2
```

Select framework 6 from upstream menu? No — this wrapper directly calls MLXServer binary.

### 5. Test

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
```

## What's Different from Upstream

| Feature | Upstream | This fork |
|---------|----------|-----------|
| Qwen3.6 Config-I support | ❌ Broken LM head loading | ✅ Fixed per-tensor lookup for MoE models |
| Thinking mode | ❌ Double-enable → 400 error | ✅ Detects prefilled `<think>` |
| KV cache scheme selection | ❌ Hardcoded | ✅ `--kv <scheme>` (e.g. `turbo4v2`) |
| Speculative prefill (experimental) | ❌ | ✅ Module present, integration pending |

## Branches

- `ek/tom-eric-moe-tuning` — our working branch (based on upstream's `ek/tom-eric-moe-tuning`)
- `main` — mirrors upstream main (do not commit here)

## Key Commits

- `4537a49` — Fix per-tensor quantization for Qwen3.6 Config-I MoE models
- `fde5fd4` — Add `--kv` CLI flag; bump swift-transformers
- `a439af0` — Resolve thinking mode double-enable conflict
- `0aee329` — Add Q-capture hook for speculative prefill (in progress)

## Updating Your Fork

```bash
# Pull latest from your fork (shivam2014/mlx-swift-lm)
./update_mlxserver_fork.sh
```
# If you want to sync with upstream ekryski changes:
git fetch origin
git rebase origin/ek/tom-eric-moe-tuning
# Resolve conflicts, then push to your fork:
git push fork ek/tom-eric-moe-tuning
```

## Experimental: Speculative Prefill

Paper: arXiv:2603.02631 — Cross-Family Speculative Prefill

Status: Engine implemented in `Sources/SpeculativePrefill/` (not yet integrated). See that module's README for details.

## Troubleshooting

**Server crashes with "Failed to load the default metallib"**  
Run `make metal` manually or ensure `./update_mlxserver_fork.sh` completed the metallib step.

**Out of memory**  
- Ensure KV scheme is `turbo4v2` or smaller
- Reduce context length
- Use a smaller model (e.g., Qwen3.5-27B instead of 35B)

**Thinking mode not working**  
Check server logs: it should detect `<think>` in the last 8 tokens of the prompt. If missing, ensure your chat template includes it.

**Model fails to load with shape mismatch**  
This fork fixes Qwen3.6 Config-I; ensure you're using the patched branch and rebuilt the server.
