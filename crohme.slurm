#!/bin/bash
#SBATCH -J new_1_TAP_weightnoise
#SBATCH -p ocr_1080
#SBATCH --qos=mid
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o new_1_TAP_weightnoise.log



python -u train_netease_noise.py
