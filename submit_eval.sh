#!/bin/bash

#SBATCH --job-name=cn_eval
#SBATCH --output=cn_eval.out.%j
#SBATCH --error=cn_eval.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=sd3
#SBATCH --partition=p5

set -ex

CONTROLNET_CKPT=$1
DATASET=$2

echo "Evaluating controlnet model: $CONTROLNET_CKPT with dataset: $DATASET"

mkdir -p outputs/$CONTROLNET_CKPT

python evaluate.py \
    --verbose True \
    --input_dataset $DATASET \
    --model "/weka/home-brianf/sd35_models/sd3.5_large.safetensors" \
    --model_folder "/weka/home-brianf/sd35_models" \
    --controlnet_ckpt $CONTROLNET_CKPT \
    --vae "/weka/applied-shared/sd3_ref/sd3_vae.safetensors" \
    --sampler "euler" \
    --text_encoder_device "cuda" \
    --out_dir "outputs/$CONTROLNET_CKPT" \