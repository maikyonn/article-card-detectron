#!/usr/bin/env python
# Refactored Detectron2 Training Script Aligned with Template

import logging
import os
import random
import json
import argparse
import yaml
import csv
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel
import cv2
import numpy as np
import wandb

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
    transforms as T,
    DatasetMapper
)
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
    DefaultPredictor
)
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    DatasetEvaluators,
    SemSegEvaluator,
    COCOPanopticEvaluator,
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    PascalVOCDetectionEvaluator,
    LVISEvaluator,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.utils.events import EventStorage, CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode, Instances
from detectron2.utils import comm

logger = logging.getLogger("detectron2")


def get_article_card_dicts(json_file):
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    dataset_dicts = []
    for img in dataset['images']:
        record = {}
        record["file_name"] = os.path.join(os.path.dirname(json_file), "..", "images", img["file_name"])
        record["image_id"] = img["id"]
        record["height"] = img["height"]
        record["width"] = img["width"]
        
        annos = [anno for anno in dataset['annotations'] if anno['image_id'] == img['id']]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(cfg, train_json, val_json):
    DatasetCatalog.register("article_card_train", lambda: get_article_card_dicts(train_json))
    DatasetCatalog.register("article_card_val", lambda: get_article_card_dicts(val_json))
    for d in ["train", "val"]:
        MetadataCatalog.get(f"article_card_{d}").set(thing_classes=["article_card"])
        MetadataCatalog.get(f"article_card_{d}").evaluator_type = "coco"


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    if evaluator_type == "cityscapes_sem_seg":
        evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
    if evaluator_type == "pascal_voc":
        evaluator_list.append(PascalVOCDetectionEvaluator(dataset_name))
    if evaluator_type == "lvis":
        evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "No Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def tile_image(image, tile_size=(1000, 1000), overlap=200):
    height, width = image.shape[:2]
    tiles = []
    for y in range(0, height, tile_size[1] - overlap):
        for x in range(0, width, tile_size[0] - overlap):
            tile = image[y:y+tile_size[1], x:x+tile_size[0]]
            if tile.shape[0] == tile_size[1] and tile.shape[1] == tile_size[0]:
                tiles.append({"image": tile, "coords": (x, y)})
    return tiles


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.stagnant_epochs = 0
        self.best_ap = 0

    def __call__(self, current_ap):
        if self.best_score is None:
            self.best_score = current_ap
            self.best_ap = current_ap
            return False
        elif current_ap > self.best_score + self.delta:
            self.best_score = current_ap
            self.best_ap = current_ap
            self.stagnant_epochs = 0
            return False
        else:
            self.stagnant_epochs += 1
            return self.stagnant_epochs >= self.patience


def do_test(cfg, model, early_stopping=None, wandb_run=None):
    results = {}
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(cfg, dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            if wandb_run:
                wandb_run.log({f"validation_{k}": v for k, v in results_i.items()})
            
            # Save results to a CSV file
            output_file = os.path.join(cfg.OUTPUT_DIR, f"{dataset_name}_validation_results.csv")
            
            # Prepare the data for CSV
            csv_data = []
            header = ['run']
            for task, metrics in results_i.items():
                for metric_name in metrics.keys():
                    header.append(f"{task}_{metric_name}")
            
            # If the file doesn't exist, create it with the header
            if not os.path.exists(output_file):
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header[1:])  # Exclude the 'run' column
            
            # Prepare the row data
            row = []
            for task, metrics in results_i.items():
                for metric_value in metrics.values():
                    row.append(metric_value)
            
            # Append the new row to the CSV file
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            logger.info(f"Appended validation results to {output_file}")
        
    
    # Early stopping based on 'bbox' AP
    if early_stopping and 'bbox' in results[dataset_name]:
        current_ap = results[dataset_name]['bbox']['AP']
        if early_stopping(current_ap):
            logger.info(f"Early stopping triggered. Best AP: {early_stopping.best_ap}")
            return True  # Signal to stop training
    return False


def do_train(cfg, model, early_stopping, wandb_run):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        WandbWriter(wandb_run)
    ] if comm.is_main_process() else []

    data_loader = build_detection_train_loader(
        cfg,
        mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN), T.RandomFlip()]),
        sampler=torch.utils.data.distributed.DistributedSampler(dataset) if comm.get_world_size() > 1 else None
    )
    if comm.is_main_process():
        logger.info("Starting training from iteration {}".format(start_iter))
    
    with EventStorage(start_iter) as storage:
        for iteration, data in zip(range(start_iter, max_iter), data_loader):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                storage.step()
                should_stop = do_test(cfg, model, early_stopping, wandb_run)
                if should_stop:
                    logger.info("Early stopping triggered.")
                    break

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
    return


class WandbWriter:
    """
    Custom writer to integrate Wandb with Detectron2's EventStorage.
    """
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run

    def write(self):
        pass  # All logging is handled manually in do_test and training loop


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    # Load base config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    
    # Now merge from your custom config file
    cfg.merge_from_file(args.config_file)
    
    # Set the data directory and output directory from the command-line arguments
    cfg.DATASETS.TRAIN = ("article_card_train",)
    cfg.DATASETS.TEST = ("article_card_val",)
    cfg.DATA_DIR = args.data_dir
    cfg.OUTPUT_DIR = args.output_dir
    
    # Add wandb_run_name to the config
    cfg.WANDB = CfgNode()
    cfg.WANDB.RUN_NAME = args.wandb_run_name
    
    # Add train and val JSON paths to the config
    cfg.TRAIN_JSON = args.train_json
    cfg.VAL_JSON = args.val_json
    
    # Set the model weights to the provided model file if it exists
    if args.model_file and os.path.exists(args.model_file):
        cfg.MODEL.WEIGHTS = args.model_file
    
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    setup_logger()
    cfg = setup(args)
    register_dataset(cfg, args.train_json, args.val_json)

    # Add debug logging
    logger.info(f"World size: {comm.get_world_size()}")
    logger.info(f"Rank: {comm.get_rank()}")
    logger.info(f"Local rank: {comm.get_local_rank()}")

    # Convert cfg to a dictionary manually
    cfg_dict = {}
    for section in dir(cfg):
        if not section.startswith('__') and not callable(getattr(cfg, section)):
            attr = getattr(cfg, section)
            if isinstance(attr, CfgNode):
                cfg_dict[section] = {k: v for k, v in attr.items()}
            else:
                cfg_dict[section] = attr

    # Initialize Wandb
    wandb_run = None
    if comm.is_main_process():
        try:
            wandb_run = wandb.init(
                project="article-detectron-final",
                name=cfg.WANDB.RUN_NAME,  # Use the run name from config
                config=cfg_dict,
                resume=False
            )
        except wandb.errors.UsageError:
            logger.warning("Failed to initialize wandb. Continuing without wandb logging.")

    # Build the model
    model = build_model(cfg)
    
    # Load model weights for fine-tuning
    if args.model_file and os.path.exists(args.model_file):
        logger.info(f"Loading model weights from: {args.model_file}")
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(args.model_file, resume=False)
    else:
        logger.info("No model file provided or file not found. Using default weights.")
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

    # Set up early stopping with hardcoded patience
    early_stopping = EarlyStopping(patience=5, delta=0.0)

    # Add this block
    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_local_rank())

    # Training
    do_train(cfg, model, early_stopping, wandb_run)

    # Final evaluation
    if comm.is_main_process():
        final_results = do_test(cfg, model, early_stopping, wandb_run)
        if wandb_run:
            wandb_run.log({"final_validation": final_results})
            wandb_run.finish()

    return


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--data-dir", required=True, help="Path to the data directory")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    parser.add_argument("--wandb-run-name", default=None, help="Name for the wandb run")
    parser.add_argument("--train-json", required=True, help="Path to the train annotations JSON file")
    parser.add_argument("--val-json", required=True, help="Path to the validation annotations JSON file")
    parser.add_argument("--model-file", default=None, help="Path to the model file for fine-tuning")
    args = parser.parse_args()
    print("Command Line Args:", args)
    
    main(args)