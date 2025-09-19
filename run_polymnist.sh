#!/bin/bash
#PBS -l select=1:ncpus=8:mem=250gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N sweep_job
#PBS -J 1-%5
#PBS -o /rds/general/user/eso18/home/mmvmvae/logs_mmvmvae/PolyMNIST/output.log
#PBS -e /rds/general/user/eso18/home/mmvmvae/logs_mmvmvae/PolyMNIST/error.log 

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate mvvae
cd $PBS_O_WORKDIR
wandb login $WANDB_API_KEY
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1

# Get params for this array index
params=$(sed -n "${PBS_ARRAY_INDEX}p" params.txt)
read dataset model seed ld lr n_ep gamma agg <<< "$params"


wandb_entity="eso18-imperial-college-london"
project_name="mvvae_polymnist"
dir_experiments="/rds/general/user/eso18/home/mmvmvae"
dir_data_base="/rds/general/user/eso18/home/mmvmvae/data/MMNIST"
dataset_name_tar="MMNIST.tar.gz"
dir_clfs_base="/rds/general/user/eso18/home/mmvmvae/trained_classifiers"
wandb_logdir="${dir_experiments}/logs_mmvmvae/PolyMNIST"
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


cd $WD
python main_mv_wsl.py \
    model=${model} \
    ++model.seed=${seed} \
    ++model.device=${device} \
    ++model.beta=${beta} \
    ++model.latent_dim=${ld} \
    ++model.lr=${lr} \
    ++model.epochs=${n_ep} \
    ++model.aggregation="${agg}" \
    dataset=${dataset} \
    ++dataset.dir_clfs_base=${dir_clfs_base} \
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