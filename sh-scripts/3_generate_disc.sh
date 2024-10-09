#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=disc_output_%j.log           
#SBATCH --time=8:00:00
#SBATCH--gres=gpu:a100:1 
#SBATCH--constraint=a100-80gb 
#SBATCH--mem-per-gpu=10GB 
#SBATCH--cpus-per-gpu=20
#SBATCH --partition=gpu

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/

python 3_generate-cards-and-comparison.py --main_dir 16k-dataset-5 --output_dir 16k-dataset-5 --use_gpu