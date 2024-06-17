#!/bin/bash

#SBATCH --account=FoMo_AIISDH
#SBATCH --partition=all_usr_prod
#SBATCH -e /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/test_conda_%j.err
#SBATCH -o /work/FoMo_AIISDH/scascianelli/2024_emuru/jobs/test_conda_%j.out
#SBATCH --nodes=1
#SBATCH --time=00:05:00


cd /work/FoMo_AIISDH/fquattrini/emuru || exit
source activate emuru

export SCRIPT=train_vae.py
export OMP_NUM_THREADS=16

srun test_conda.py