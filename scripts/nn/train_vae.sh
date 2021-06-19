#!/usr/bin/env bash


python scripts/nn/train_og_ddp.py --beta 4 \
    --lr 1e-4 \
    --z_dim 20 \
    --objective "H" \
    --model "H" \
    --max_iter 10000 \
    --display_step 1000 \
    --save_step 1000 \
    --num_workers 4 \
    --dataset "train_ddp_1_exit_22x40_doors" \
    --ckpt_name "vae" \
    --ckpt_name_og "ddp" \
    "train_vae"
    # "test_vae"
    # "test_ddp"
    # "train_ddp"
