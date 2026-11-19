#!/bin/bash
# Seeds 43 and 44 of the CIFAR-10 operating points of Table 1 (ResNet-18 lambda = 0.09, ResNet-34 0.13,
# ResNet-50 0.11, ResNet-152 0.2) and of the dense baselines of all five ResNets. The ResNet-101 seeds
# come from search/cifar10_sweep.py --seeds. Three queues, one per GPU; finished runs are skipped.
#
#   GPUS=0,1,2 bash scripts/run_cifar10_seeds.sh
set -u
cd "$(dirname "$0")/.."
IFS=',' read -ra G <<< "${GPUS:-0,1,2}"
OUT=results/cifar10_seeds

run() {  # model lambda per-GPU-batch seed
  local d=$OUT/$1_cifar10_lam$2_seed$4
  [ -f $d/result.json ] && return
  mkdir -p $d
  torchrun --nproc_per_node=1 --master_port=$PORT trainers/train_resnet.py --model $1 --dataset cifar10 \
    --lambda_sparsity $2 --epochs 50 --finetune_epochs 15 --per_gpu_batch $3 --seed $4 --run_dir $d > $d/driver.log 2>&1
}

(export CUDA_VISIBLE_DEVICES=${G[0]} PORT=31043
 run resnet152 0.2 64 43; run resnet152 0.0 64 43; run resnet101 0.0 128 43; run resnet34 0.0 128 43
 run resnet18 0.0 128 43) &
(export CUDA_VISIBLE_DEVICES=${G[1]} PORT=31044
 run resnet152 0.2 64 44; run resnet152 0.0 64 44; run resnet101 0.0 128 44; run resnet34 0.0 128 44
 run resnet18 0.0 128 44; run resnet50 0.11 128 43) &
(export CUDA_VISIBLE_DEVICES=${G[2]} PORT=31045
 run resnet18 0.09 128 43; run resnet18 0.09 128 44; run resnet34 0.13 128 43; run resnet34 0.13 128 44
 run resnet50 0.0 128 43; run resnet50 0.0 128 44; run resnet50 0.11 128 44) &
wait
