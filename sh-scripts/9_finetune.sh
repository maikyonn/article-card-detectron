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
python 9_finetune.py --config-file model-config/article_card_config.yaml     --data-dir data/v4-coco/     --output-dir data/v4-coco-finetune-20kiter     --wandb-run-name v1-t50-finetune-20kiter     --train-json data/v4-coco/annotations/train_annotations.json     --val-json data/v4-coco/annotations/val_annotations.json     --model-file data/v4-coco-finetune/model_final.pth