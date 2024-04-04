#!/bin/bash

#SBATCH --account=FoMo_AIISDH
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/train_vae_%j.err
#SBATCH -o /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/train_vae_%j.out
#SBATCH --mem=24G
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%1

source ~/anaconda3/etc/profile.d/conda.sh

cd /work/FoMo_AIISDH/fquattrini/emuru || exit
source activate emuru

export SCRIPT=train_vae.py
export OMP_NUM_THREADS=16


if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  export SCRIPT_ARGS=" \
      --mixed_precision bf16 \
      --output_dir /work/FoMo_AIISDH/scascianelli/2024_emuru/results \
      --logging_dir /work/FoMo_AIISDH/scascianelli/2024_emuru/results \
      --train_batch_size 32 \
      --htr_path /work/FoMo_AIISDH/scascianelli/2024_emuru/results/8da9/model_1000 \
      --writer_id_path /work/FoMo_AIISDH/scascianelli/2024_emuru/results/b12a/model_2900 \
      --run_id 1b7f \
      "
  accelerate launch --num_processes 4 $SCRIPT "$SCRIPT_ARGS"
fi

# all other cases
if [[ "$SLURM_ARRAY_TASK_ID" != "0" ]]; then
  export SCRIPT_ARGS=" \
    --mixed_precision bf16 \
    --output_dir /work/FoMo_AIISDH/scascianelli/2024_emuru/results \
    --logging_dir /work/FoMo_AIISDH/scascianelli/2024_emuru/results \
    --train_batch_size 32 \
    --resume_id 1b7f \
    "
  accelerate launch --num_processes 4 $SCRIPT "$SCRIPT_ARGS"
fi