#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=endea_output_%j.log           
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/

python 3_generate-cards-and-comparison.py --main_dir 16k-dataset-5 --output_dir 16k-dataset-5 --use_gpu