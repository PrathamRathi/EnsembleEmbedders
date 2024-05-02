#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=24g
#SBATCH -n 4
#SBATCH -o log/ccv_%j_$1_$2_$3_$4.out

module load python/3.9.16s-x3wdtvt 

source .venv/bin/activate

python main.py -e $1 -lr $2 -n $3 -p $4
#SBATCH -o log/ccv_%j_$1_$2_$3_$4.out