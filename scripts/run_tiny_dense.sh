#!/bin/bash
# Dense (lambda = 0) Tiny-ImageNet baselines of Table 1: seed 42, three GPUs with synchronized batch
# normalization, and the epochs and batch of the pruned run of the same depth. Finished runs are skipped.
#
#   GPUS=0,1,2 bash scripts/run_tiny_dense.sh
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2}

run() {  # model fine-tuning-epochs per-GPU-batch run-dir
  [ -f $4/result.json ] && return
  mkdir -p $4
  torchrun --nproc_per_node=3 --master_port=31850 trainers/train_resnet.py --model $1 --dataset tiny_imagenet \
    --lambda_sparsity 0.0 --epochs 50 --finetune_epochs $2 --per_gpu_batch $3 --seed 42 --run_dir $4 > $4/driver.log 2>&1
}

run resnet18  55 128 results/tiny_dense_matched/resnet18_tiny_dense
run resnet34  65 128 results/tiny_dense_matched/resnet34_tiny_dense
run resnet50  55 128 results/tiny_dense_matched/resnet50_tiny_dense
run resnet101 55 128 results/tiny_dense_r101/dense_seed42
run resnet152 55 64  results/tiny_dense_matched/resnet152_tiny_dense
