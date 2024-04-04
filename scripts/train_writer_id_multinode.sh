#!/bin/bash

#SBATCH --account=FoMo_AIISDH
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/gasampling_%j.err
#SBATCH -o /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/gasampling_%j.out
#SBATCH --mem=24G
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -J writer_id_training
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%1

######################
### Set enviroment ###
######################
source ~/anaconda3/etc/profile.d/conda.sh
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $SLURM_NTASKS \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "
export PYTHON_FILE="train_writer_id.py"

cd /work/FoMo_AIISDH/fquattrini/emuru || exit
source activate emuru

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
export OMP_NUM_THREADS=8
export PYTHON_ARGS=" \
    --mixed_precision bf16 \
    --output_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --logging_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --train_batch_size 512 \
    --lr 1e-2 \
    --run_id d93b \
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $PYTHON_ARGS"
srun $CMD
fi

if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
export OMP_NUM_THREADS=8
export SCRIPT_ARGS=" \
    --mixed_precision bf16 \
    --output_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --logging_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --train_batch_size 512 \
    --lr 1e-2 \
    --resume_id d93b \
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"
srun $CMD
fi

if [[ "$SLURM_ARRAY_TASK_ID" == "2" ]]; then
export OMP_NUM_THREADS=8
export SCRIPT_ARGS=" \
    --mixed_precision bf16 \
    --output_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --logging_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --train_batch_size 512 \
    --lr 1e-2 \
    --resume_id d93b \
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"
srun $CMD
fi

if [[ "$SLURM_ARRAY_TASK_ID" == "3" ]]; then
export OMP_NUM_THREADS=8
export SCRIPT_ARGS=" \
    --mixed_precision bf16 \
    --output_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --logging_dir /work/FoMo_AIISDH/scascianelli/2024_emuru \
    --train_batch_size 512 \
    --lr 1e-2 \
    --resume_id d93b \
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"
srun $CMD
fi