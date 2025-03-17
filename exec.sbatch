#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J AdvGA
#SBATCH -t 720:00:00
#SBATCH -o out2.out
#SBATCH -e err2.err
#SBATCH --partition=gpu-ampere


conda activate cs234_a3
echo $CONDA_PREFIX

ml load cuda/12.5

python main.py --sys_name "Advection2D" --run_num 4 --cases "ga" --seed 1
