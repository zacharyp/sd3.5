#!/bin/bash

#SBATCH --job-name=cn_eval
#SBATCH --output=cn_eval.out.%j
#SBATCH --error=cn_eval.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=sd3
#SBATCH --partition=p5

set -ex

CONTROL_TYPE=$1

CKPT_PATH=/weka/home-brianf/comfy-models/controlnet/${CONTROL_TYPE}_8b.safetensors
DATASET=/weka/home-brianf/controlnet_eval/${CONTROL_TYPE}_eval_pkl/pkl
OUT_DIR=outputs/controlnet_8b_${CONTROL_TYPE}

echo "ckpt path: $CKPT_PATH, control type: $CONTROL_TYPE, dataset: $DATASET, out_dir: $OUT_DIR"

mkdir -p $OUT_DIR

python evaluate.py \
    --input_dataset $DATASET \
    --model "/weka/home-brianf/sd35_models/sd3.5_large.safetensors" \
    --model_folder "/weka/home-brianf/sd35_models" \
    --controlnet_ckpt $CKPT_PATH \
    --vae "/weka/applied-shared/sd3_ref/sd3_vae.safetensors" \
    --sampler "euler" \
    --text_encoder_device "cuda" \
    --out_dir $OUT_DIR \