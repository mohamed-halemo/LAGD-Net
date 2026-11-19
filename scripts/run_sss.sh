#!/bin/bash
# SSS baseline of Table 2 on ResNet-101, under the recipe and epoch budget of the LAGD-Net runs.
# gamma = 1.75 * lambda / N, with lambda of the LAGD-Net operating point and N = 33 blocks.
#
#   GPUS=0,1,2 bash scripts/run_sss.sh cifar10        # seeds 42-44, gamma = 0.35/33, 65 epochs, one GPU each
#   GPUS=0,1,2 bash scripts/run_sss.sh tiny_imagenet  # seed 42, gamma = 0.7/33, 105 epochs, three GPUs
#
# Each part ends with the export benchmark. On Tiny-ImageNet, the dense, SSS and LAGD-Net networks
# are timed together, and Table 2 scales the LAGD-Net speedup of Table 1 by their latency ratio.
set -u
cd "$(dirname "$0")/.."
IFS=',' read -ra G <<< "${GPUS:-0,1,2}"
OUT=results/sss
mkdir -p $OUT

case "${1:-}" in
cifar10)
  gamma=$(python3 -c "print(0.35 / 33)")
  for i in 0 1 2; do
    seed=$((42 + i)); d=$OUT/sss_g0.35_seed$seed
    [ -f $d/result.json ] || { mkdir -p $d; CUDA_VISIBLE_DEVICES=${G[$i]} torchrun --nproc_per_node=1 \
      --master_port=$((31740 + i)) trainers/train_sss.py --dataset cifar10 --model resnet101 --gamma $gamma \
      --epochs 65 --seed $seed --run_dir $d > $d/driver.log 2>&1; } &
  done
  wait
  CUDA_VISIBLE_DEVICES=${G[0]} python3 benchmark/benchmark.py --out $OUT/bench_g0.35.json \
    --runs $OUT/sss_g0.35_seed42 $OUT/sss_g0.35_seed43 $OUT/sss_g0.35_seed44
  ;;
tiny_imagenet)
  gamma=$(python3 -c "print(0.7 / 33)")
  d=$OUT/tiny_g0.7_seed42
  [ -f $d/result.json ] || { mkdir -p $d; CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2} torchrun --nproc_per_node=3 \
    --master_port=31750 trainers/train_sss.py --dataset tiny_imagenet --model resnet101 --gamma $gamma \
    --epochs 105 --seed 42 --run_dir $d > $d/driver.log 2>&1; }
  CUDA_VISIBLE_DEVICES=${G[0]} python3 benchmark/benchmark.py --out $OUT/bench_tiny_joint.json \
    --runs results/tiny_dense_r101/dense_seed42 $d results/resnet_101_152/resnet101_attempt03
  ;;
*)
  echo "usage: GPUS=0,1,2 bash scripts/run_sss.sh cifar10|tiny_imagenet"
  exit 1
  ;;
esac
