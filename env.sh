# Source in every shell and job script:  source env.sh
# Caches stay on /mnt/hdd2; credentials load from outside the repo (never commit them).
export D3_ROOT=/mnt/hdd2/Abhijit/projects/D3
export D3_PY=$D3_ROOT/.venv/bin/python
export D3_OUT=$D3_ROOT/outputs                       # run outputs (gitignored)
export HF_HOME=/mnt/hdd2/Abhijit/.cache/huggingface
export UV_CACHE_DIR=/mnt/hdd2/Abhijit/.cache/uv
export PIP_CACHE_DIR=/mnt/hdd2/Abhijit/.cache/pip
export TORCH_HOME=/mnt/hdd2/Abhijit/.cache/torch
export TRITON_CACHE_DIR=/mnt/hdd2/Abhijit/.cache/triton
export VLLM_CACHE_ROOT=/mnt/hdd2/Abhijit/.cache/vllm
export TMPDIR=/mnt/hdd2/Abhijit/.cache/tmp
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}   # Abhijit: D3 runs use GPUs 0 and 1 only
mkdir -p "$D3_OUT" "$TMPDIR"
if [ -f ~/.config/secrets/research.env ]; then set -a; . ~/.config/secrets/research.env; set +a; fi
export HF_HOME=/mnt/hdd2/Abhijit/.cache/huggingface   # research.env must not override the project cache
