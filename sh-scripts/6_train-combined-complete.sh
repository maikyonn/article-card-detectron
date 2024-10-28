#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=end_compl-v3_%j.log           
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --partition=isi
module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/

export OMP_NUM_THREADS=16
python 6_train-detectron.py \
    --config-file model-config/nms_article_card_config.yaml \
    --data-dir ./data/v3-coco-complete \
    --output-dir ./runs/v3-complete-output \
    --wandb-run-name v3-complete
    --num-gpus 1