#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=end_v3-v2t50_%j.log           
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=isi
module load git
source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /home1/spangher/miniconda3/envs/layout

cd /project/jonmay_1426/spangher/homepage-parser-latest/

export OMP_NUM_THREADS=16
python 6_train-detectron.py \
    --config-file model-config/nms_article_card_config.yaml \
    --data-dir ./data/v3-coco-v1t50 \
    --output-dir ./runs/v3-v2t50 \
    --wandb-run-name v3-v2t50
    --num-gpus 1