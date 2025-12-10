#!/bin/bash
#PBS -l select=1:ncpus=8:mem=250gb:ngpus=1
#PBS -l walltime=40:00:00
#PBS -N celeba_sweep
#PBS -J 1-45
#PBS -o /rds/general/user/eso18/home/mmvmvae/run_celeba/logs/output.log
#PBS -e /rds/general/user/eso18/home/mmvmvae/run_celeba/logs/error.log 

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate mvvae
cd $PBS_O_WORKDIR
wandb login $WANDB_API_KEY
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Get params for this array index
params=$(sed -n "${PBS_ARRAY_INDEX}p" params/celeba_params.txt) 
read dataset model seed agg alpha <<< "$params"

# local wandb instance
wandb_entity="eso18-imperial-college-london"
project_name="mvvae_celeba"
dir_experiments="/rds/general/user/eso18/home/mmvmvae"
dir_data_base="/rds/general/user/eso18/home/mmvmvae/data/CelebA"
dataset_name_tar="CelebA.tar.gz"
# dir_clfs_base="/rds/general/user/eso18/home/mmvmvae/trained_classifiers"
wandb_logdir="${dir_experiments}/run_celeba/logs"
WD=$(pwd)
device="cuda"  # 'cuda' if you are useing a GPU
log_freq_downstream=50
log_freq_coherence=50
log_freq_lhood=50
log_freq_plotting=50
ld=256
beta=1      # KL weight (float)
aa=true     # alpha annealing steps (int)
a_w=0.5
ld=512 
lr=5e-4 
n_ep=400 
gamma=0.0001
cov_scalar=$(echo "1 - ($alpha^2)" | bc -l)

cd $WD
python main_mv_wsl.py \
    model=${model} \
    ++model.seed=${seed} \
    ++model.alpha_scalar=${alpha} \
    ++model.cov_scalar=${cov_scalar} \
    ++model.device=${device} \
    ++model.beta=${beta} \
    ++model.latent_dim=${ld} \
    ++model.lr=${lr} \
    ++model.epochs=${n_ep} \
    ++model.aggregation="${agg}" \
    dataset=${dataset} \
    ++dataset.modalities_order=${mod} \
    ++log.wandb_offline=false \
    ++log.downstream_logging_frequency=${log_freq_downstream} \
    ++log.coherence_logging_frequency=${log_freq_coherence} \
    ++log.likelihood_logging_frequency=${log_freq_lhood} \
    ++log.img_plotting_frequency=${log_freq_plotting} \
    ++log.wandb_offline=false \
    ++log.wandb_local_instance=true \
    ++log.wandb_entity=${wandb_entity} \
    ++log.wandb_group="full_runs" \
    ++log.wandb_project_name=${project_name} \
    ++log.dir_logs=${wandb_logdir}