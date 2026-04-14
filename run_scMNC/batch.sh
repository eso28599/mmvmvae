#!/bin/bash
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=00:20:00
#PBS -N sc_1_5
#PBS -J 1-90
#PBS -o run_scMNC/logs/output.log
#PBS -e run_scMNC/logs/error.log 
eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate mvvae

# define user specific variables
cd mmvmvae
export folder_path=${PWD}
project_name="mvvae_scMNC" # project name in wandb
# Set environment variables
cd $PBS_O_WORKDIR
wandb login $WANDB_API_KEY
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Get params for this array index
params=$(sed -n "${PBS_ARRAY_INDEX}p" run_scMNC/scMNC_params.txt) 
# params=$(sed -n "${PBS_ARRAY_INDEX}p" run_scMNC/indiv.txt) 
read model seed agg alpha <<< "$params"

# experimental running variables
log_freq=1
wandb_logdir="${folder_path}/run_scMNC/logs"
device="cuda"  # 'cuda' if you are using a GPU

# model specific variables
aa=true  # whether to use alpha annealing for mixedprior
cov_scalar=$(echo "1 - ($alpha^2)" | bc -l) # covariance scalar for jointprior

# arrchitecture variables
ld=32  # latent dimension

# training variables, not investigated
lr=1e-3  # learning rate (changed from 5e-4)
n_ep=150 # number of epochs
batch_size=32 # batch size

# training variables, investigated
early_stop=false # whether to use early stopping
beta_anneal=true # whether to use beta/kl annealing
beta_M=20 # frequency of beta increase

python run_experiment.py \
    model=${model} \
    ++model.seed=${seed} \
    ++model.alpha_scalar=${alpha} \
    ++model.cov_scalar=${cov_scalar} \
    ++model.device=${device} \
    ++model.beta=${beta} \
    ++model.batch_size=${batch_size} \
    ++model.latent_dim=${ld} \
    ++model.alpha_annealing=${aa} \
    ++model.early_stop=${early_stop} \
    ++model.beta_annealing=${beta_anneal} \
    ++model.beta_M=${beta_M} \
    ++model.lr=${lr} \
    ++model.epochs=${n_ep} \
    ++model.aggregation="${agg}" \
    dataset="scMNC" \
    ++log.downstream_logging_frequency=${log_freq} \
    ++log.coherence_logging_frequency=${log_freq} \
    ++log.likelihood_logging_frequency=${log_freq} \
    ++log.img_plotting_frequency=${log_freq} \
    ++log.val_freq=${log_freq} \
    ++log.wandb_offline=false \
    ++log.wandb_local_instance=true \
    ++log.wandb_group="full_runs" \
    ++log.wandb_project_name=${project_name} \
    ++log.dir_logs=${wandb_logdir}