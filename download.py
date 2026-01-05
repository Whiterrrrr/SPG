import os
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(
    repo_id='GSAI-ML/LLaDA-8B-Instruct',
    local_dir='/data/discrete-diffusionRL-LLMhf_models/LLaDA-8B-Instruct',
    local_dir_use_symlinks=False,
    resume_download=True
)