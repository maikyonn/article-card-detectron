#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=end_16k-t100_%j.log           
#SBATCH --time=48:00:00
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
    --config-file article_card_config.yaml \
    --data-dir ./16k-coco-t100 \
    --output-dir ./16k-coco-t100-output-v2 \
    --wandb-run-name t100-nms-official
    --num-gpus 1