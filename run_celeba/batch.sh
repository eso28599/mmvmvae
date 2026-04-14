#!/bin/bash
#PBS -l select=1:ncpus=8:mem=250gb:ngpus=1
#PBS -l walltime=10:00:00
#PBS -N celeba_batch
#PBS -J 41-45
#PBS -o run_celeba/logs/output.log
#PBS -e run_celeba/logs/error.log 

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate mvvae

# define user specific variables
cd mmvmvae
export folder_path=${PWD}
project_name="mvvae_celeba" # project name in wandb

# Set environment variables
cd $PBS_O_WORKDIR
wandb login $WANDB_API_KEY
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Get params for this array index
params=$(sed -n "${PBS_ARRAY_INDEX}p" run_celeba/celeba_params.txt) 
## to specify specific indices use the below
# IDX=$(sed -n "${PBS_ARRAY_INDEX}p" run_celeba/indices.txt)
# params=$(sed -n "${IDX}p" run_celeba/celeba_params.txt) 
read dataset model seed agg alpha <<< "$params"

# experimental running variables
log_freq=1
img_log_freq=50
wandb_logdir="${folder_path}/run_celeba/logs"
device="cuda"  # 'cuda' if you are using a GPU

# model specific variables
aa=true  # whether to use alpha annealing for mixedprior
cov_scalar=$(echo "1 - ($alpha^2)" | bc -l) # covariance scalar for jointprior

# arrchitecture variables
ld=512  # latent dimension

# training variables
lr=5e-4  # learning rate

# # og paper
# ld=128  # latent dimension
# lr=2e-4  # learning rate

n_ep=400 # number of epochs
early_stop=true # whether to use early stopping
beta_anneal=true # whether to use beta/kl annealing

python run_experiment.py \
    model=${model} \
    ++model.seed=${seed} \
    ++model.alpha_scalar=${alpha} \
    ++model.cov_scalar=${cov_scalar} \
    ++model.device=${device} \
    ++model.beta=${beta} \
    ++model.latent_dim=${ld} \
    ++model.alpha_annealing=${aa} \
    ++model.early_stop=${early_stop} \
    ++model.beta_annealing=${beta_anneal} \
    ++model.lr=${lr} \
    ++model.epochs=${n_ep} \
    ++model.aggregation="${agg}" \
    dataset=${dataset} \
    ++dataset.modalities_order=${mod} \
    ++log.downstream_logging_frequency=${log_freq} \
    ++log.coherence_logging_frequency=${log_freq} \
    ++log.likelihood_logging_frequency=${log_freq} \
    ++log.img_plotting_frequency=${img_log_freq} \
    ++log.val_freq=${log_freq} \
    ++log.wandb_offline=false \
    ++log.wandb_local_instance=true \
    ++log.wandb_group="full_runs" \
    ++log.wandb_project_name=${project_name} \