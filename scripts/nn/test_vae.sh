#!/usr/bin/env bash


python scripts/nn/train_og_ddp.py --beta 4 \
    --lr 1e-4 \
    --z_dim 20 \
    --objective "H" \
    --model "H" \
    --max_iter 20 \
    --display_step 20 \
    --save_step 20 \
    --num_workers 4 \
    --dataset "test_ddp_1_exit_22x40_doors" \
    --ckpt_name "vae" \
    --ckpt_name_og "dpp" \
    --batch_size 1 \
    "test_vae"
