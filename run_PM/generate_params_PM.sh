#!/bin/bash
# generate_params_PM.sh

# common file names
dataset_names=("PMtranslated75")
seeds=(1 2 3 4 5)
models=("joint" "mixedprior" "unimodal" "jointprior")
outfile=run_PM/pm_params.txt 
> $outfile
outfile_mod=run_PM/pm_params_mod.txt > $outfile_mod

for dataset in "${dataset_names[@]}"; do
for model in "${models[@]}"; do
if [ "$model" = "joint" ]; then 
  aggregation_fs=("avg" "moe" "mopoe")
  alphas=(0)
elif [ "$model" = "jointprior" ]; then
  aggregation_fs=("avg")
  alphas=(0.1 0.3 0.5 0.7 0.9 0.93 0.95 0.97 0.99)
else
  aggregation_fs=("avg")
  alphas=(0)
fi
for seed in "${seeds[@]}"; do
for agg in "${aggregation_fs[@]}"; do
for alpha in "${alphas[@]}"; do
if (( $(echo "$alpha == 0.97" | bc -l) )); then
  modalities=(0 1 2 3 4)
else
  modalities=(0)
fi
for mod in "${modalities[@]}"; do
    if [ "$mod" -eq 0 ]; then 
      echo "$dataset $model $seed $agg $alpha $mod" >> $outfile
    else 
      echo "$dataset $model $seed $agg $alpha $mod" >> $outfile_mod
    fi
done
done
done
done
done
done

cat run_PM/pm_params.txt run_PM/pm_params_mod.txt > run_PM/pm_all.txt