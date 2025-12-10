#!/bin/bash
# generate_params_celeba.sh

# common file names
dataset_names=("CelebA")
seeds=(1 2 3 4 5)
models=("joint" "mixedprior" "jointprior")
outfile=run_celeba/celeba_params.txt 
> $outfile

for dataset in "${dataset_names[@]}"; do
for model in "${models[@]}"; do
if [ "$model" = "joint" ]; then 
  aggregation_fs=("moe")
  alphas=(0)
elif [ "$model" = "jointprior" ]; then
  aggregation_fs=("avg")
  alphas=(0.1 0.5 0.9 0.93 0.95 0.97 0.99)
else
  aggregation_fs=("avg")
  alphas=(0)
fi
for seed in "${seeds[@]}"; do
for agg in "${aggregation_fs[@]}"; do
for alpha in "${alphas[@]}"; do
    echo "$dataset $model $seed $agg $alpha" >> $outfile
done
done
done
done
done