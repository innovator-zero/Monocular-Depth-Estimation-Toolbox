export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/dinov2/lib

torchrun --nproc_per_node=1  \
    tools/train.py configs/vit/vitb16_nyu_linear_config.py  --launcher pytorch