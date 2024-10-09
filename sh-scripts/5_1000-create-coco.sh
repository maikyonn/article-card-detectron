#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=endea_output_%j.log           
#SBATCH --time=40:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=isi

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/

python 5_create-coco.py --downloads_dir 16k-dataset-5 --dataset_dir 16k-coco-t1000 --top_collections_file top_1000_collections.txt 