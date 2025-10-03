#!/bin/bash
# generate_params.sh

# common file names
dataset_names=("PMtranslated75")
seeds=(1 2 3 4 5) # repeats
gammas=(0.0001)
alphas=(0.1 0.3 0.5 0.7 0.9 0.93 0.95 0.97 0.99)
latent_dims=(512) # not changed
alpha_annealing=(true)
n_epochs=(400)
learning_rates=(5e-4)
log_freq_downstream=50
log_freq_coherence=50
log_freq_lhood=50
log_freq_plotting=50


models=("joint") 
aggregation_fs=("avg" "moe" "mopoe") 
outfile=params.txt 
> $outfile

for dataset in "${dataset_names[@]}"; do
for model in "${models[@]}"; do
for seed in "${seeds[@]}"; do
for lr in "${learning_rates[@]}"; do
for ld in "${latent_dims[@]}"; do
for n_ep in "${n_epochs[@]}"; do
for gamma in "${gammas[@]}"; do
for agg in "${aggregation_fs[@]}"; do
    echo "$dataset $model $seed $ld $lr $n_ep $gamma $agg 0" >> $outfile
done
done
done
done
done
done
done
done


models=("mixedprior" "unimodal")
aggregation_fs=("avg") 
outfile=params_mixed_uni.txt
> $outfile

for dataset in "${dataset_names[@]}"; do
for model in "${models[@]}"; do
for seed in "${seeds[@]}"; do
for lr in "${learning_rates[@]}"; do
for ld in "${latent_dims[@]}"; do
for n_ep in "${n_epochs[@]}"; do
for gamma in "${gammas[@]}"; do
for agg in "${aggregation_fs[@]}"; do
    echo "$dataset $model $seed $ld $lr $n_ep $gamma $agg 0" >> $outfile
done
done
done
done
done
done
done
done


models=("jointprior")
outfile=params_jp.txt
> $outfile

for dataset in "${dataset_names[@]}"; do
for model in "${models[@]}"; do
for seed in "${seeds[@]}"; do
for lr in "${learning_rates[@]}"; do
for ld in "${latent_dims[@]}"; do
for n_ep in "${n_epochs[@]}"; do
for gamma in "${gammas[@]}"; do
for agg in "${aggregation_fs[@]}"; do
for alpha in "${alphas[@]}"; do
    echo "$dataset $model $seed $ld $lr $n_ep $gamma $agg $alpha" >> $outfile
done
done
done
done
done
done
done
done
done