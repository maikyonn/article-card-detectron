#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=endea_output_%j.log           
#SBATCH --time=40:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=isi

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/

cp -rv 16k-dataset-5 16k-dataset-v1-backup