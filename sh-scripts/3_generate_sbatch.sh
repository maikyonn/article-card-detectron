#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=endea_output_%j.log           
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH--mem=32GB
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=isi

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout-2

cd /project/jonmay_1426/spangher/homepage-parser-latest && python 3_generate-cards-and-comparison.py --main_dir data/v2-datasets-playwright/new-dataset-16k/ --output_dir data/v2-datasets-playwright/new-dataset-16k/ --use_gpu