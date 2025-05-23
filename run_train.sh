#!/bin/bash

python -m train.train_mdm  \
    --save_dir save/openasl.test.v0.35k \
    --dataset openasl \
    --cond_mask_prob 0 \
    --lambda_rcxyz 0 \
    --lambda_vel 1 \
    --lambda_fc 0 \
    --batch_size 128 \
    --gen_guidance_param 1 \
    --eval_during_training \
    --arch gru  \
    --overwrite 



