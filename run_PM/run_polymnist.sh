#!/bin/bash
#PBS -l select=1:ncpus=8:mem=250gb:ngpus=1
#PBS -l walltime=30:00:00
#PBS -N sweep_01234
#PBS -J 1-90
#PBS -o run_PM/logs/output.log
#PBS -e run_PM/logs/error.log 

# -J 1-90 for pm_all.txt to run all PM experimentsß
# -J 1-20 for pm_params_mod.txt to run modalities PM experiments
# -J 1-70 for pm_params.txt to run main PM experiments

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate mvvae

# define user specific variables
cd mmvmvae
export folder_path=${PWD}
project_name="mvvae_polymnist" # project name in wandb

# Set environment variables
cd $PBS_O_WORKDIR
wandb login $WANDB_API_KEY
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Get params for this array index
params=$(sed -n "${PBS_ARRAY_INDEX}p" run_PM/pm_all.txt)
# params=$(sed -n "${PBS_ARRAY_INDEX}p" run_PM/pm_params.txt) # main experiment
# params=$(sed -n "${PBS_ARRAY_INDEX}p" run_PM/pm_params_mod.txt) # mod experiment
read dataset model seed agg alpha mod <<< "$params"
cd $(pwd)

# experimental logging variables
log_freq=50
wandb_logdir="${folder_path}/run_PM/logs"
device="cuda"  # 'cuda' if you are useing a GPU

# model specific variables
aa=true  # whether to use alpha annealing for mixedprior
cov_scalar=$(echo "1 - ($alpha^2)" | bc -l) # covariance scalar for jointprior

# architecture variables
ld=512  # latent dimension

# training specific variables
lr=5e-4  # learning rate
n_ep=400 # number of epochs
early_stop=false # whether to use early stopping
beta_anneal=true # whether to use beta/kl annealing

python main_mv_wsl.py \
    model=${model} \
    ++model.seed=${seed} \
    ++model.alpha_scalar=${alpha} \
    ++model.cov_scalar=${cov_scalar} \
    ++model.device=${device} \
    ++model.latent_dim=${ld} \
    ++model.lr=${lr} \
    ++model.epochs=${n_ep} \
    ++model.aggregation="${agg}" \
    dataset=${dataset} \
    ++dataset.modalities_order=${mod} \
    ++log.downstream_logging_frequency=${log_freq} \
    ++log.coherence_logging_frequency=${log_freq} \
    ++log.likelihood_logging_frequency=${log_freq} \
    ++log.img_plotting_frequency=${log_freq} \
    ++log.wandb_offline=false \
    ++log.wandb_local_instance=true \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_group="full_runs" \
    ++log.wandb_project_name=${project_name} \
    ++log.dir_logs=${wandb_logdir}