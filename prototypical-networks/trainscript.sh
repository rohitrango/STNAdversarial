#!/bin/bash

python scripts/train/few_shot/run_train.py --run_stn 1 \
    --model.stn_model stnvae \
    --train.stn_reg_coeff 0.1 \
    --data.cuda

