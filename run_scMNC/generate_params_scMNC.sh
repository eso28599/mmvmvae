#!/bin/bash
# generate_params_scMNC.sh

# common file names
seeds=(1 2 3 4 5)
models=("joint" "mixedprior" "jointprior")
outfile=run_scMNC/scMNC_params.txt 
> $outfile

for model in "${models[@]}"; do
if [ "$model" = "joint" ]; then 
  aggregation_fs=("moe")
  alphas=(0)
elif [ "$model" = "jointprior" ]; then
  aggregation_fs=("avg")
  alphas=(0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.93 0.95 0.97 0.99)
else
  aggregation_fs=("avg")
  alphas=(0)
fi
for seed in "${seeds[@]}"; do
for agg in "${aggregation_fs[@]}"; do
for alpha in "${alphas[@]}"; do
    echo "$model $seed $agg $alpha" >> $outfile
done
done
done
done


seeds=(1 2 3 4 5)
models=("jointprior")
aggregation_fs=("avg")
alphas=(0.97)
beta_anneals=(true false)
# over 150 epochs, this corresponds to 1.5 and 3 epochs to increase beta respectively, 
# usually it converges in 20 epochs, so this corresponds to a very fast and fast anneal respectively
beta_Ms=(100 50 25 10 5)
outfile=run_scMNC/scMNC_params_97.txt
> $outfile


for model in "${models[@]}"; do
for seed in "${seeds[@]}"; do
for latent_dim in "${latent_dims[@]}"; do
for beta_anneal in "${beta_anneals[@]}"; do
for beta_M in "${beta_Ms[@]}"; do
for batch in "${batch_size[@]}"; do
for lr in "${lrs[@]}"; do
for alpha in "${alphas[@]}"; do
    echo "$model $seed $agg $alpha $latent_dim $beta_anneal $beta_M $batch $lr" >> $outfile
done
done
done
done
done
done
done
done