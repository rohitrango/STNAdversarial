#!/bin/bash

python scripts/train/few_shot/run_train.py --run_stn 1 \
    --model.stn_model stnvae \
    --model.hid_dim 64 \
    --data.dataset miniimagenet \
    --data.split ravi \
    --data.shot 1 \
    --train.stn_reg_coeff 1 \
    --model.x_dim 3,84,84 \
    --log.exp_dir results/stnvae \
    --train.decay_every 20 \
    --train.stn_reg_coeff 0.01 \
    --data.cuda

