pip install torch==2.5.0 torchvision torchaudio packaging wheel
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install ninja
pip install flash-attn --no-build-isolation

pip install -e .

# train model with GRPO
bash scripts/qwen25vl3b_cs_clevrsat3k_grpo.sh