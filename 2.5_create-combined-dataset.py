"""
Snapshot Comparison and Dataset Creation Tool
-------------------------------------------

This script compares pairs of website snapshots (old vs new) using YOLO object detection
to determine which snapshot has better quality, then creates a combined dataset with the
best versions.

Features:
- Multi-GPU support for parallel processing
- Automatic handling of Windows/Unix path differences
- Supports processing specific subfolders
- Preserves directory structure in output
- Copies associated CSV files from the 'new' directory
- Configurable file suffixes for images and data files

Usage:
    python 2.5_create-combined-dataset.py --old <old_dir> --new <new_dir> --output <output_dir> [options]

Required Arguments:
    --old       Path to directory containing the old snapshots
    --new       Path to directory containing the new snapshots
    --output    Path to directory where the combined dataset will be saved

Optional Arguments:
    --subfolder       Specific subfolder to process (processes all if not specified)
    --model          Path to YOLO model file (default: yolov8s.pt)
    --img-suffix     Suffix for image files (default: '.fullpage.jpg')
    --data-suffix    Suffix for data files (default: '.csv')
    --new-suffix     Suffix added to new images (default: '-new')
    --old-suffix     Suffix added to old images in output (default: '-old-best')
    --new-best-suffix Suffix added to new images in output (default: '-new-best')

Example:
    python 2.5_create-combined-dataset.py \
        --old "v5-proxy" \
        --new "v5-proxy" \
        --output "v5-proxy-combined" \
        --model "yolov8x.pt" \
        --img-suffix ".fullpage.jpg" \
        --data-suffix ".json"

Notes:
- Automatically handles filename sanitization for Windows systems
- Requires CUDA-compatible GPU(s)
- Uses up to 4 GPUs if available
- Data files are always copied from the 'new' directory
"""

import os
import sys
import shutil
import logging
import argparse
import multiprocessing
from PIL import Image

from ultralytics import YOLO
import pandas as pd
import torch

# Global variable to hold the YOLO model in each process
yolo_model = None

def configure_logging(gpu_id):
    """
    Configures logging for each process.
    """
    logger = logging.getLogger()
    logger.handlers = []  # Remove any existing handlers
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f'%(asctime)s - [GPU {gpu_id}] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def run_yolo_detection(image_path, logger):
    """
    Runs YOLO detection on the given image and returns the number of detections.
    """
    try:
        logger.info(f"Processing image {image_path}")
        # Force CPU inference for MPS compatibility
        results = yolo_model.predict(
            source=image_path, 
            imgsz=16384, 
            max_det=1000, 
            verbose=False,
            device='cpu'  # Force CPU for prediction to avoid MPS limitations
        )
        num_detections = len(results[0].boxes)
        logger.info(f"Detected {num_detections} objects in {image_path}")
        return num_detections
    except Exception as e:
        logger.error(f"Error during YOLO detection on {image_path}: {e}")
        return 0

def sanitize_filename(filename, replace_colon):
    """
    Conditionally sanitizes the filename by replacing colons with hyphens if required.
    """
    if replace_colon:
        return filename.replace(':', '-')
    return filename

def is_windows():
    """
    Checks if the current operating system is Windows.
    """
    return os.name == 'nt'

def get_image_size(image_path):
    """
    Retrieves the width and height of the image using Pillow.
    """
    with Image.open(image_path) as img:
        return img.width, img.height

def process_image_pair(args):
    """
    Processes a single image pair (old and new images).
    """
    (file, old_root, new_root, output_dir, replace_colon, old_dir, new_dir, logger, suffixes) = args
    try:
        base_name = file[:-len(suffixes['img'])]
        sanitized_base_name = sanitize_filename(base_name, replace_colon)

        old_image_path = os.path.join(old_root, file)
        new_image_name = f"{sanitized_base_name}{suffixes['new']}{suffixes['img']}"
        new_image_path = os.path.join(new_root, new_image_name)

        if not os.path.exists(new_image_path):
            logger.warning(f"New image not found for {old_image_path}. Skipping.")
            return

        # Run YOLO on old image
        old_count = run_yolo_detection(old_image_path, logger)

        # Run YOLO on new image
        new_count = run_yolo_detection(new_image_path, logger)

        # Determine better snapshot
        if old_count > new_count * 1.2:
            best_image_path = old_image_path
            best_suffix = suffixes['old_best']
        else:
            best_image_path = new_image_path
            best_suffix = suffixes['new_best']

        # Always use the CSV from 'new'
        csv_source_path = new_image_path.replace(suffixes['img'], suffixes['data'])

        # Define output paths
        if best_image_path.startswith(old_dir):
            relative_best_path = os.path.relpath(best_image_path, old_dir)
        else:
            relative_best_path = os.path.relpath(best_image_path, new_dir)

        sanitized_relative_best_path = sanitize_filename(relative_best_path, replace_colon)
        best_image_name = os.path.basename(sanitized_relative_best_path)

        # Correct naming for output files
        if best_image_name.endswith(suffixes['new'] + suffixes['img']):
            best_image_name = best_image_name.replace(suffixes['new'] + suffixes['img'], f'{best_suffix}{suffixes['img']}')
        elif best_image_name.endswith(suffixes['img']):
            best_image_name = best_image_name.replace(suffixes['img'], f'{best_suffix}{suffixes['img']}')
        else:
            best_image_name += f'{best_suffix}{suffixes['img']}'  # Fallback

        # Construct the full output image path
        output_image_path = os.path.join(output_dir, os.path.dirname(sanitized_relative_best_path), best_image_name)
        output_csv_path = output_image_path.replace(suffixes['img'], suffixes['data'])

        # Ensure the output subdirectory exists
        output_image_dir = os.path.dirname(output_image_path)
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
            logger.info(f"Created subdirectory: {output_image_dir}")

        # Copy the best image
        shutil.copy2(best_image_path, output_image_path)
        logger.info(f"Copied best image to {output_image_path}")

        # Copy the CSV from 'new' directory
        if os.path.exists(csv_source_path):
            shutil.copy2(csv_source_path, output_csv_path)
            logger.info(f"Copied CSV from {csv_source_path} to {output_csv_path}")
        else:
            logger.warning(f"CSV file not found: {csv_source_path}")

        logger.info(f"Processed {base_name}: Saved best image ({best_suffix}) and CSV to {output_image_path}")
    except Exception as e:
        logger.error(f"Error processing file {file}: {e}")

def gather_image_pairs(old_dir, new_dir, replace_colon, img_suffix, new_suffix, specific_subfolder=None):
    """
    Gathers all image pairs to be processed.
    """
    image_pairs = []
    new_img_suffix = f"{new_suffix}{img_suffix}"

    if specific_subfolder:
        old_subfolder = os.path.join(old_dir, specific_subfolder)
        new_subfolder = os.path.join(new_dir, specific_subfolder)
        if not os.path.exists(old_subfolder) or not os.path.exists(new_subfolder):
            print(f"Specified subfolder '{specific_subfolder}' not found in both old and new directories.")
            return image_pairs
        for file in os.listdir(old_subfolder):
            if file.endswith(img_suffix) and not file.endswith(new_img_suffix):
                image_pairs.append((file, old_subfolder, new_subfolder, replace_colon, old_dir, new_dir))
    else:
        for root, dirs, files in os.walk(old_dir):
            relative_path = os.path.relpath(root, old_dir)
            new_root = os.path.join(new_dir, relative_path)
            for file in files:
                if file.endswith(img_suffix) and not file.endswith(new_img_suffix):
                    image_pairs.append((file, root, new_root, replace_colon, old_dir, new_dir))

    return image_pairs

def get_device(gpu_id=None):
    """
    Determines the appropriate device to use (CUDA, MPS, or CPU).
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return f'cuda:{gpu_id}' if gpu_id is not None else 'cuda'
    return 'cpu'

def worker(gpu_id, tasks, model_path, output_dir, replace_colon, old_dir, new_dir, suffixes):
    """
    Worker function for each GPU process to handle assigned image pairs.
    """
    logger = configure_logging(gpu_id)
    
    try:
        global yolo_model
        yolo_model = YOLO(model_path)
        device = get_device(gpu_id)
        yolo_model.to(device)
        logger.info(f"Initialized YOLO model on device: {device}")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {e}")
        sys.exit(1)

    for task in tasks:
        file, old_root, new_root, replace_colon_task, old_dir_task, new_dir_task = task
        process_image_pair((file, old_root, new_root, output_dir, replace_colon, old_dir, new_dir, logger, suffixes))

def main():
    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Compare snapshots using YOLO object detection across multiple GPUs.")
    parser.add_argument("--old", required=True, help="Path to the 'old' directory")
    parser.add_argument("--new", required=True, help="Path to the 'new' directory")
    parser.add_argument("--output", required=True, help="Path to the output directory")
    parser.add_argument("--subfolder", help="Specific subfolder to process")
    parser.add_argument("--model", default='yolov8s.pt', help="Path to the YOLO model file")
    parser.add_argument("--img-suffix", default='.fullpage.jpg', help="Suffix for image files")
    parser.add_argument("--data-suffix", default='.csv', help="Suffix for data files")
    parser.add_argument("--new-suffix", default='-new', help="Suffix added to new images")
    parser.add_argument("--old-suffix", default='-old-best', help="Suffix added to old images in output")
    parser.add_argument("--new-best-suffix", default='-new-best', help="Suffix added to new images in output")
    args = parser.parse_args()

    # Determine if sanitization is needed based on OS
    replace_colon_in_filenames = is_windows()

    # Configure main logger
    main_logger = configure_logging("Main")

    if replace_colon_in_filenames:
        main_logger.info("Operating System detected: Windows. Colons in filenames will be replaced with hyphens.")
    else:
        main_logger.info("Operating System detected: Unix-like. Colons in filenames will be retained.")

    # Check for available compute devices
    if torch.backends.mps.is_available():
        main_logger.info("Apple Silicon GPU detected. Using MPS backend.")
        num_gpus_to_use = 1  # MPS only supports single GPU
    elif torch.cuda.is_available():
        num_gpus_available = torch.cuda.device_count()
        num_gpus_to_use = min(4, num_gpus_available)
        main_logger.info(f"Detected {num_gpus_available} CUDA GPUs. Utilizing {num_gpus_to_use} GPUs for processing.")
    else:
        main_logger.error("No GPU detected. This script requires GPU acceleration. Exiting.")
        sys.exit(1)

    # Gather all image pairs
    image_pairs = gather_image_pairs(args.old, args.new, replace_colon_in_filenames, args.img_suffix, args.new_suffix, args.subfolder)
    if not image_pairs:
        main_logger.error("No image pairs found to process. Exiting.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        main_logger.info(f"Created output directory: {args.output}")

    # Distribute image pairs across GPUs
    gpu_ids = list(range(num_gpus_to_use))
    tasks_per_gpu = [[] for _ in gpu_ids]
    for idx, pair in enumerate(image_pairs):
        gpu_id = gpu_ids[idx % num_gpus_to_use]
        tasks_per_gpu[gpu_id].append(pair)

    # Start separate processes for each GPU
    processes = []
    for gpu_id, tasks in zip(gpu_ids, tasks_per_gpu):
        if not tasks:
            main_logger.info(f"No tasks assigned to GPU {gpu_id}. Skipping.")
            continue
        p = multiprocessing.Process(target=worker, args=(gpu_id, tasks, args.model, args.output, replace_colon_in_filenames, args.old, args.new, {
            'img': args.img_suffix,
            'data': args.data_suffix,
            'new': args.new_suffix,
            'old_best': args.old_suffix,
            'new_best': args.new_best_suffix
        }))
        p.start()
        processes.append(p)
        main_logger.info(f"Started process on GPU {gpu_id} with {len(tasks)} image pairs.")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    main_logger.info("All image pairs have been processed.")

if __name__ == "__main__":
    main()