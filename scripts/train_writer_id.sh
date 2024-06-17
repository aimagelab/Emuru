#!/bin/bash

#SBATCH --account=FoMo_AIISDH
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/scascianelli/2024_bmvc_alpha/jobs/sampling_%j.err
#SBATCH -o /work/FoMo_AIISDH/scascianelli/2024_bmvc_alpha/jobs/sampling_%j.out
#SBATCH --mem=24G
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-13