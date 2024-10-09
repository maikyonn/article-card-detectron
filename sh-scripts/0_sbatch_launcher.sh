#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=discovery_output_%j.log           
#SBATCH --time=8:00:00

#SBATCH --partition=gpu

module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/
s3cmd sync s3://ml-datasets-maikyon/16k-dataset-2/ ./16k-dataset-5/