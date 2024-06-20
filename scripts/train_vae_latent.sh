#!/bin/bash

#SBATCH --account=FoMo_AIISDH
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/train_htr_%j.err
#SBATCH -o /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/train_htr_%j.out
#SBATCH --mem=42G
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --nodes=1
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -J htr_training
#SBATCH --array=0-5%1

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate emuru

cd /work/FoMo_AIISDH/fquattrini/Emuru

export OMP_NUM_THREADS=16
export SCRIPT=train_htr.py

export SCRIPT_ARGS=" \
    --output_dir /work/FoMo_AIISDH/scascianelli/2024_emuru/results_htr \
    --logging_dir /work/FoMo_AIISDH/scascianelli/2024_emuru/results_htr \
    --train_batch_size 128 \
    --resume_id bbb0 \
    "

if [[ $SLURM_ARRAY_TASK_ID  -eq 0 ]]; then
    /homes/$(whoami)/.conda/envs/emuru/bin/accelerate launch --num_processes 1 $SCRIPT $SCRIPT_ARGS

elif [[ $SLURM_ARRAY_TASK_ID  -eq 1 ]]; then
    /homes/$(whoami)/.conda/envs/emuru/bin/accelerate launch --num_processes 1 $SCRIPT $SCRIPT_ARGS

elif [[ $SLURM_ARRAY_TASK_ID  -eq 2 ]]; then
    /homes/$(whoami)/.conda/envs/emuru/bin/accelerate launch --num_processes 1 $SCRIPT $SCRIPT_ARGS

elif [[ $SLURM_ARRAY_TASK_ID  -eq 3 ]]; then
    /homes/$(whoami)/.conda/envs/emuru/bin/accelerate launch --num_processes 1 $SCRIPT $SCRIPT_ARGS

elif [[ $SLURM_ARRAY_TASK_ID  -eq 4 ]]; then
    /homes/$(whoami)/.conda/envs/emuru/bin/accelerate launch --num_processes 1 $SCRIPT $SCRIPT_ARGS

elif [[ $SLURM_ARRAY_TASK_ID  -eq 5 ]]; then
    /homes/$(whoami)/.conda/envs/emuru/bin/accelerate launch --num_processes 1 $SCRIPT $SCRIPT_ARGS

fi