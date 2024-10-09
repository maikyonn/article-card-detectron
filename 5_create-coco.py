#!/usr/bin/env python3
"""
convert_to_coco.py

This script converts a directory of collections with images and CSV annotations into the COCO format for Detectron2.
It supports splitting the dataset into training and validation sets. Additionally, it allows processing only a subset
of collections specified in a text file (e.g., the top 500 collections).

Usage:
    python convert_to_coco.py 
        --downloads_dir /path/to/downloads_directory 
        --dataset_dir /path/to/dataset_directory 
        --top_collections_file top_500_collections.txt 
        [--images_dir_name images] 
        [--annotations_dir_name annotations] 
        [--train_annotations_file train_annotations.json] 
        [--val_annotations_file val_annotations.json] 
        [--categories object] 
        [--split_ratio 0.8] 
        [--random_seed 42]

Dependencies:
    - pandas
    - argparse
    - pathlib
    - tqdm
    - PIL
    - sklearn
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse
import logging

def setup_logging():
    """
    Configures the logging settings to log information to both file and console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler('convert_to_coco.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert downloads folder to COCO format with train/val split for Detectron2 training, processing only specified top collections.")

    parser.add_argument(
        '--downloads_dir',
        type=str,
        default='16k-dataset-5',
        help='Path to the downloads directory. *(Default: downloads-15k)*'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='16k-coco',
        help='Path to the target dataset directory. *(Default: dataset-15k)*'
    )
    parser.add_argument(
        '--images_dir_name',
        type=str,
        default='images',
        help='Name of the images subdirectory within dataset_dir. *(Default: images)*'
    )
    parser.add_argument(
        '--annotations_dir_name',
        type=str,
        default='annotations',
        help='Name of the annotations subdirectory within dataset_dir. *(Default: annotations)*'
    )
    parser.add_argument(
        '--train_annotations_file',
        type=str,
        default='train_annotations.json',
        help='Name of the training COCO annotations JSON file. *(Default: train_annotations.json)*'
    )
    parser.add_argument(
        '--val_annotations_file',
        type=str,
        default='val_annotations.json',
        help='Name of the validation COCO annotations JSON file. *(Default: val_annotations.json)*'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['article-card'],  # Modify if you have multiple classes
        help='List of category names. *(Default: object)*'
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.9,
        help='Proportion of the dataset to include in the training set. *(Default: 0.8)*'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Seed used by the random number generator. *(Default: 42)*'
    )
    # New Argument: Path to Top Collections Text File
    parser.add_argument(
        '--top_collections_file',
        type=str,
        default="top_500_collections.txt",
        help='Path to the text file containing names of top collections to process (e.g., top_500_collections.txt).'
    )

    return parser.parse_args()

def read_top_collections(top_collections_file):
    """
    Reads the top collections names from a text file.

    Args:
        top_collections_file (str): Path to the text file containing collection names.

    Returns:
        set: A set of collection names.
    """
    try:
        with open(top_collections_file, 'r') as f:
            # Assuming one collection name per line
            collections = set(line.strip() for line in f if line.strip())
        logging.info(f"Read {len(collections)} collection names from '{top_collections_file}'.")
        return collections
    except Exception as e:
        logging.error(f"Failed to read top collections file '{top_collections_file}': {e}")
        return set()

def convert_downloads_to_coco_with_split(
    downloads_dir='downloads-15k',
    dataset_dir='dataset-15k',
    images_dir_name='images',
    annotations_dir_name='annotations',
    train_annotations_file='train_annotations.json',
    val_annotations_file='val_annotations.json',
    categories=['article-card'],  # Modify if you have multiple classes
    split_ratio=0.8,
    random_seed=42,
    top_collections_file=None  # New Parameter
):
    """
    Convert a directory of collections with images and CSV annotations to COCO format for Detectron2,
    splitting the dataset into training and validation sets. Optionally, process only specified top collections.

    Args:
        downloads_dir (str): Path to the source downloads directory.
        dataset_dir (str): Path to the target dataset directory.
        images_dir_name (str): Name of the images subdirectory within dataset_dir.
        annotations_dir_name (str): Name of the annotations subdirectory within dataset_dir.
        train_annotations_file (str): Name of the training COCO annotations JSON file.
        val_annotations_file (str): Name of the validation COCO annotations JSON file.
        categories (list): List of category names.
        split_ratio (float): Proportion of the dataset to include in the training set.
        random_seed (int): Seed used by the random number generator.
        top_collections_file (str): Path to the text file containing top collection names to process.
    """
    # Define paths
    downloads_path = Path(downloads_dir)
    dataset_path = Path(dataset_dir)
    images_path = dataset_path / images_dir_name
    annotations_path = dataset_path / annotations_dir_name
    train_annotations_json = annotations_path / train_annotations_file
    val_annotations_json = annotations_path / val_annotations_file

    # Create target directories
    images_path.mkdir(parents=True, exist_ok=True)
    annotations_path.mkdir(parents=True, exist_ok=True)

    # Initialize COCO structures for train and val
    coco_train = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    coco_val = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create category mapping
    category_map = {name: idx for idx, name in enumerate(categories, start = 0)}
    for name, id in category_map.items():
        coco_train['categories'].append({
            "id": id,
            "name": name,
            "supercategory": "none"
        })
        coco_val['categories'].append({
            "id": id,
            "name": name,
            "supercategory": "none"
        })

    annotation_id = 1
    image_id = 1
    image_records = []  # To store image metadata for splitting

    # Read top collections if provided
    top_collections = set()
    if top_collections_file:
        top_collections = read_top_collections(top_collections_file)
        if not top_collections:
            logging.error("Top collections set is empty. Exiting.")
            return

    # Iterate through each collection folder
    if top_collections:
        # Only process collections in the top_collections set
        collection_folders = [downloads_path / name for name in top_collections]
        # Verify that these directories exist
        existing_collections = [p for p in collection_folders if p.exists() and p.is_dir()]
        missing_collections = top_collections - set(p.name for p in existing_collections)
        if missing_collections:
            logging.warning(f"{len(missing_collections)} collections listed in '{top_collections_file}' were not found in '{downloads_dir}'. They will be skipped.")
    else:
        # If no top_collections_file is provided, process all collections
        collection_folders = [p for p in downloads_path.iterdir() if p.is_dir()]
        logging.info(f"Found {len(collection_folders)} collection(s) in '{downloads_dir}'.")

    logging.info(f"Processing {len(collection_folders)} collection(s).")

    for collection in tqdm(collection_folders, desc="Processing collections"):
        collection_name = collection.name

        # Skip if not in top_collections (redundant if top_collections is used to filter)
        if top_collections and collection_name not in top_collections:
            continue

        # Iterate through each image in the collection
        image_files = list(collection.glob('*.jpg')) + list(collection.glob('*.jpeg')) + list(collection.glob('*.png'))
        for image_file in image_files:
            # Determine the base name by removing the '.fullpage' suffix if present
            base_name = image_file.stem
            if base_name.endswith('.fullpage'):
                base_name = base_name.replace('.fullpage', '')
            else:
                # If there's no '.fullpage' suffix, use the stem as is
                pass  # Modify if you have other suffixes

            # Corresponding CSV file
            csv_file = image_file.parent / f"{base_name}.csv"
            if not csv_file.exists():
                logging.warning(f"CSV annotation for image '{image_file.name}' not found in collection '{collection_name}'. Skipping image.")
                continue

            # Read CSV annotations
            try:
                df = pd.read_csv(csv_file)
                # Validate required columns
                required_columns = {'x', 'y', 'width', 'height'}
                if not required_columns.issubset(df.columns):
                    logging.warning(f"CSV file '{csv_file}' is missing required columns {required_columns}. Skipping image.")
                    continue
            except Exception as e:
                logging.error(f"Error reading CSV file '{csv_file}' in collection '{collection_name}': {e}. Skipping image.")
                continue

            # Open image to get dimensions
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception as e:
                logging.error(f"Error opening image file '{image_file}' in collection '{collection_name}': {e}. Skipping image.")
                continue

            # Store image metadata for splitting
            image_records.append({
                "image_file": image_file,
                "csv_file": csv_file,
                "width": width,
                "height": height,
                "image_id": image_id
            })
            image_id += 1

    if not image_records:
        logging.error("No valid image records found. Exiting.")
        return

    # Split the dataset into train and val
    train_records, val_records = train_test_split(
        image_records,
        train_size=split_ratio,
        random_state=random_seed,
        shuffle=True
    )

    logging.info(f"Total images: {len(image_records)}")
    logging.info(f"Training set: {len(train_records)} images")
    logging.info(f"Validation set: {len(val_records)} images")

    # Function to copy images and add to COCO
    def process_records(records, coco, split_name):
        nonlocal annotation_id
        for record in tqdm(records, desc=f"Processing {split_name} records"):
            image_file = record["image_file"]
            csv_file = record["csv_file"]
            width = record["width"]
            height = record["height"]
            image_id_current = record["image_id"]

            # Define target image path
            target_image_path = images_path / image_file.name

            # Check if the image already exists in the target directory
            if not target_image_path.exists():
                # Copy image to target images directory
                try:
                    shutil.copy2(image_file, target_image_path)
                except Exception as e:
                    logging.error(f"Failed to copy image '{image_file}' to '{target_image_path}': {e}. Skipping image.")
                    continue
            else:
                logging.info(f"Image '{target_image_path.name}' already exists in the target directory. Skipping copy.")

            # Add image info to COCO
            coco['images'].append({
                "id": image_id_current,
                "file_name": target_image_path.name,
                "height": height,
                "width": width
            })

            # Process each bounding box without any validation
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                logging.error(f"Error reading CSV file '{csv_file}' during COCO processing: {e}. Skipping annotations for this image.")
                continue

            for _, row in df.iterrows():
                x, y, w, h = row['x'], row['y'], row['width'], row['height']

                # Assign category_id (assuming single category as per 'categories' list)
                category_id = 0  # Modify if you have multiple categories

                # Add annotation
                coco['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id_current,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1

    # Process training records
    logging.info("\nProcessing Training Records...")
    process_records(train_records, coco_train, "training")

    # Process validation records
    logging.info("\nProcessing Validation Records...")
    process_records(val_records, coco_val, "validation")

    # Save COCO annotations to JSON
    try:
        with open(train_annotations_json, 'w') as f:
            json.dump(coco_train, f, indent=4)
        logging.info(f"Successfully saved training COCO annotations to '{train_annotations_json}'.")
    except Exception as e:
        logging.error(f"Failed to save training COCO annotations to '{train_annotations_json}': {e}")

    try:
        with open(val_annotations_json, 'w') as f:
            json.dump(coco_val, f, indent=4)
        logging.info(f"Successfully saved validation COCO annotations to '{val_annotations_json}'.")
    except Exception as e:
        logging.error(f"Failed to save validation COCO annotations to '{val_annotations_json}': {e}")

    logging.info(f"All images have been copied to '{images_path}'.")
    logging.info("Dataset preparation with split is complete.")

if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    convert_downloads_to_coco_with_split(
        downloads_dir=args.downloads_dir,
        dataset_dir=args.dataset_dir,
        images_dir_name=args.images_dir_name,
        annotations_dir_name=args.annotations_dir_name,
        train_annotations_file=args.train_annotations_file,
        val_annotations_file=args.val_annotations_file,
        categories=args.categories,
        split_ratio=args.split_ratio,
        random_seed=args.random_seed,
        top_collections_file=args.top_collections_file  # Pass the new argument
    )