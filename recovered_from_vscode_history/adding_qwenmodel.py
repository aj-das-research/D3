from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct',
    local_dir='./qwen-model',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*']
)