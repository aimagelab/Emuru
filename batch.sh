#!/bin/bash
# python train_T5.py --wandb --resume_dir files/checkpoints/Emuru_100k_FW --resume --output_dir files/checkpoints/Emuru_100k_FW_tune_0.60 --start_alpha 0.60 --num_train_epochs 6
python train_T5.py --wandb --resume_dir files/checkpoints/Emuru_100k_FW --resume --output_dir files/checkpoints/Emuru_100k_FW_tune_0.70 --start_alpha 0.70 --num_train_epochs 6
python train_T5.py --wandb --resume_dir files/checkpoints/Emuru_100k_FW --resume --output_dir files/checkpoints/Emuru_100k_FW_tune_0.80 --start_alpha 0.80 --num_train_epochs 6
python train_T5.py --wandb --resume_dir files/checkpoints/Emuru_100k_FW --resume --output_dir files/checkpoints/Emuru_100k_FW_tune_0.95 --start_alpha 0.95 --num_train_epochs 6
python train_T5.py --wandb --resume_dir files/checkpoints/Emuru_100k_FW --resume --output_dir files/checkpoints/Emuru_100k_FW_tune_0.99 --start_alpha 0.99 --num_train_epochs 6
python train_T5.py --wandb --resume_dir files/checkpoints/Emuru_100k_FW --resume --output_dir files/checkpoints/Emuru_100k_FW_tune_0.999 --start_alpha 0.999 --num_train_epochs 6
python train_T5.py --wandb --resume_dir files/checkpoints/Emuru_100k_FW --resume --output_dir files/checkpoints/Emuru_100k_FW_tune_1.0 --start_alpha 1.0 --num_train_epochs 6