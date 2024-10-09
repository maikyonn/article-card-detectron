#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=endeavour_output_%j.log           
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/
s3cmd sync s3://ml-datasets-maikyon/16k-dataset/ ./16k-dataset-3/