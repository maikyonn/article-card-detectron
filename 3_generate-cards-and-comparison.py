"""
OCR Processing and Validation Tool

This script processes collections of images, performs OCR on specified regions, and validates
the extracted text against provided JSON data. It's designed to handle large datasets of
document images with corresponding bounding box information.

Key Features:
- Crops images based on bounding box coordinates from CSV files
- Performs OCR on cropped regions using EasyOCR
- Validates OCR results against expected text from JSON files
- Saves cropped images and OCR results in organized directory structure
- Supports multiple languages and GPU acceleration
- Includes progress tracking and comprehensive logging

Usage:
    python script.py --main_dir INPUT_DIR --output_dir OUTPUT_DIR [options]

Required Arguments:
    --main_dir      Path to directory containing collection subdirectories
    --output_dir    Path where results will be saved

Optional Arguments:
    --ocr_output_suffix   Suffix for OCR results CSV (default: "_ocr_results.csv")
    --image_extension     Extension for input images (default: ".fullpage.jpg")
    --csv_extension      Extension for CSV files (default: ".csv")
    --html_extension     Extension for HTML files (default: ".html")
    --categories_present Flag if CSV contains category column
    --languages         OCR languages, space-separated (default: "en")
    --use_gpu          Flag to enable GPU acceleration

Input Directory Structure:
    main_dir/
    ├── collection1/
    │   ├── image1.fullpage.jpg
    │   ├── image1.fullpage.csv
    │   └── image1.hyperlinks.json
    └── collection2/
        └── ...

Output Directory Structure:
    output_dir/
    ├── collection1/
    │   ├── box-images/
    │   │   └── cropped_images.jpg
    │   └── collection1_ocr_results.csv
    └── collection2/
        └── ...

CSV Format:
    Required columns: x, y, width, height
    Optional columns: category

Notes:
    - JSON files should contain text entries to validate against OCR results
    - The similarity score (0.0 to 1.0) indicates how well the JSON text is contained
      within the OCR text, with 1.0 meaning perfect containment
    - Processing can be resumed as it skips collections with existing results
"""

import os
import random
import cv2
import pandas as pd
from pathlib import Path
import argparse
import easyocr
from PIL import Image
from tqdm import tqdm
import logging
import json
from difflib import SequenceMatcher
import time
import shutil

def setup_logging():
    """
    Configures the logging settings.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler('processing.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

def perform_ocr(reader, image_path, languages=['en']):
    """
    Perform OCR on the given image and return the extracted text and confidence scores.

    Parameters:
    - reader (easyocr.Reader): Initialized EasyOCR reader.
    - image_path (str): Path to the image file.
    - languages (list): List of languages to support. Default is English.

    Returns:
    - ocr_texts (list): List of extracted texts.
    - ocr_confidences (list): List of confidence scores corresponding to the texts.
    """
    try:
        results = reader.readtext(image_path)
        ocr_texts = [result[1] for result in results]
        ocr_confidences = [result[2] for result in results]
    except Exception as e:
        logging.error(f"OCR processing failed for {image_path}: {e}")
        ocr_texts = []
        ocr_confidences = []
    return ocr_texts, ocr_confidences

def read_json_data(json_path):
    """
    Read JSON data from a local file.

    Parameters:
    - json_path (Path): Path to the JSON file.

    Returns:
    - data (list): List of dictionaries containing JSON data.
    """
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        logging.info(f"Successfully read JSON data from {json_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to read JSON data from {json_path}: {e}")
        return []

def calculate_similarity(ocr_text, json_text):
    """
    Calculate if json_text is contained within ocr_text, with fuzzy matching.
    Returns 1.0 if json_text is fully contained, 0.0 if not found at all.
    
    Parameters:
    - ocr_text (str): The longer text from OCR
    - json_text (str): The shorter text we're looking for
    """
    # Normalize both texts
    ocr_text = ocr_text.lower().strip()
    json_text = json_text.lower().strip()
    
    if not json_text or not ocr_text:  # Handle empty strings
        return 0.0
        
    # Split into words to handle word-level matching
    json_words = json_text.split()
    ocr_words = ocr_text.split()
    
    # Count how many words from json_text are found in ocr_text
    words_found = sum(1 for word in json_words 
                     if any(SequenceMatcher(None, word, ocr_word).ratio() > 0.8 
                           for ocr_word in ocr_words))
    
    # Calculate percentage of json words found
    return words_found / len(json_words) if json_words else 0.0

def find_best_match(ocr_text, json_texts):
    """
    Find the best matching text in json_texts for the given ocr_text.
    A perfect score (1.0) means the json text is fully contained within the OCR text.
    """
    best_score = 0.0
    best_match_text = ""
    for json_entry in json_texts:
        json_text = json_entry.get('text', '')
        score = calculate_similarity(ocr_text, json_text)
        if score > best_score:
            best_score = score
            best_match_text = json_text
    return best_score, best_match_text

def process_collection(
    collection_path,
    output_dir,
    ocr_output_suffix="_ocr_results.csv",
    image_extension=".fullpage.jpg",
    csv_extension=".fullpage.csv",
    html_extension=".html",
    categories_present=True,
    languages=['en'],
    use_gpu=True
):
    """
    Process a single collection: crop images based on CSV bounding boxes, perform OCR,
    validate against JSON data, and save the results.

    Args:
        collection_path (Path): Path object pointing to the collection directory.
        output_dir (Path): Path object pointing to the main output directory.
        ocr_output_suffix (str): Suffix for the OCR results CSV file.
        image_extension (str): Extension of the full-page image files.
        csv_extension (str): Extension of the CSV files containing bounding boxes.
        html_extension (str): Extension of the HTML files (not used but part of the triplet).
        categories_present (bool): Indicates if the CSV contains a 'category' column.
        languages (list): Languages for EasyOCR.
        use_gpu (bool): Whether to use GPU for EasyOCR. Set to False if no GPU is available.

    Returns:
        None
    """
    try:
        collection_name = collection_path.name
        logging.info(f"Starting processing for collection: {collection_name}")

        # Prepare output subdirectories
        collection_output = output_dir / collection_name
        collection_output.mkdir(parents=True, exist_ok=True)

        # Define the OCR results CSV path
        ocr_csv_path = collection_output / f"{collection_name}{ocr_output_suffix}"

        # **New Change: Check if OCR results already exist**
        if ocr_csv_path.exists():
            logging.info(f"OCR results already exist for collection '{collection_name}'. Skipping this collection.")
            return  # Skip processing this collection

        # Create 'box-images' directory within the collection's output directory
        box_images_output = collection_output / "box-images"
        box_images_output.mkdir(parents=True, exist_ok=True)

        # Initialize EasyOCR Reader
        try:
            reader = easyocr.Reader(languages, gpu=use_gpu)
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR Reader for collection '{collection_name}': {e}")
            return

        # Initialize list to collect OCR results for this collection
        ocr_results_list = []

        # Iterate through all CSV files in the collection directory
        csv_files = list(collection_path.glob(f"*{csv_extension}"))
        if not csv_files:
            logging.warning(f"No CSV files found in collection '{collection_name}'. Skipping.")
            return

        for csv_file in csv_files:
            # Remove '.fullpage' from the base name, but keep '-new' if present
            base_name = csv_file.stem.replace('.fullpage', '')

            # Corresponding image and JSON files
            image_file = collection_path / f"{base_name}{image_extension}"
            json_file = collection_path / f"{base_name.replace('-new', '')}.hyperlinks.json"

            if not image_file.exists():
                logging.warning(f"Image file '{image_file}' does not exist. Skipping CSV '{csv_file.name}'.")
                continue

            if not json_file.exists():
                logging.warning(f"JSON file '{json_file}' does not exist. Proceeding without JSON validation.")
                json_texts = []
            else:
                # Read JSON data from local file
                json_data = read_json_data(json_file)
                json_texts = json_data if isinstance(json_data, list) else []

            # Read the image
            image = cv2.imread(str(image_file))
            if image is None:
                logging.error(f"Failed to load image '{image_file}'. Skipping CSV '{csv_file.name}'.")
                continue

            # Read the CSV file
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                logging.error(f"Failed to read CSV file '{csv_file}': {e}. Skipping.")
                continue

            # Check if required columns exist
            required_columns = {'x', 'y', 'width', 'height'}
            if not required_columns.issubset(df.columns):
                logging.error(f"CSV file '{csv_file}' is missing required columns {required_columns}. Skipping.")
                continue

            # If categories are present, ensure the 'category' column exists
            if categories_present and 'category' not in df.columns:
                logging.warning(f"'category' column not found in '{csv_file}'. Setting category as 'object'.")
                df['category'] = 'object'

            # Iterate through each bounding box
            for idx, row in df.iterrows():
                try:
                    x = int(row['x'])
                    y = int(row['y'])
                    w = int(row['width'])
                    h = int(row['height'])
                except Exception as e:
                    logging.error(f"Invalid bounding box values in '{csv_file}' at row {idx}: {e}. Skipping.")
                    continue

                # Ensure bounding box is within image boundaries
                x = max(x, 0)
                y = max(y, 0)
                w = max(w, 1)
                h = max(h, 1)
                x_end = min(x + w, image.shape[1])
                y_end = min(y + h, image.shape[0])

                # Crop the image
                cropped_image = image[y:y_end, x:x_end]
                if cropped_image.size == 0:
                    logging.warning(f"Empty crop for '{image_file}' with bbox ({x}, {y}, {w}, {h}). Skipping.")
                    continue

                # Determine category name
                if categories_present:
                    category_name = str(row['category']).replace(" ", "_")
                else:
                    category_name = "object"

                # Construct the cropped image filename
                cropped_filename = f"{base_name}_box{idx+1}_{category_name}.jpg"
                cropped_output_path = box_images_output / cropped_filename

                # Save the cropped image
                try:
                    success = cv2.imwrite(str(cropped_output_path), cropped_image)
                    if success:
                        logging.info(f"Saved cropped image to '{cropped_output_path}'")
                    else:
                        logging.error(f"Failed to save cropped image to '{cropped_output_path}'. Skipping OCR.")
                        continue  # Skip OCR if image saving failed
                except Exception as e:
                    logging.error(f"Exception occurred while saving cropped image '{cropped_output_path}': {e}")
                    continue

                # Perform OCR on the cropped image
                ocr_texts, ocr_confidences = perform_ocr(reader, str(cropped_output_path), languages)
                extracted_text = " ".join(ocr_texts) if ocr_texts else ""
                average_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0

                # Validate OCR text against JSON data and calculate similarity score
                if json_texts:
                    similarity_score, matched_text = find_best_match(extracted_text, json_texts)
                else:
                    similarity_score = 0.0  # No JSON data to compare
                    matched_text = ""

                # Append OCR results to the list with similarity score
                ocr_results_list.append({
                    'cropped_image': cropped_filename,
                    'extracted_text': extracted_text,
                    'average_confidence': average_confidence,
                    'similarity_score': similarity_score,
                    'matched_text': matched_text
                })

        # After processing all CSVs in the collection, save OCR results to a CSV file
        if ocr_results_list:
            ocr_df = pd.DataFrame(ocr_results_list)
            try:
                ocr_df.to_csv(ocr_csv_path, index=False)
                logging.info(f"Saved OCR results to '{ocr_csv_path}'")
            except Exception as e:
                logging.error(f"Failed to save OCR results CSV for collection '{collection_name}': {e}")
        else:
            logging.warning(f"No OCR results to save for collection '{collection_name}'.")

    except Exception as e:
        logging.error(f"Exception occurred while processing collection '{collection_name}': {e}")

def create_cropped_images_and_perform_ocr(
    main_dir,
    output_dir,
    ocr_output_suffix="_ocr_results.csv",
    image_extension=".fullpage.jpg",
    csv_extension=".fullpage.csv",
    html_extension=".html",
    categories_present=True,
    languages=['en'],
    use_gpu=True
):
    """
    Traverse the main directory and process each collection sequentially to extract bounding boxes,
    crop the corresponding areas from the full-page images, perform OCR on the crops,
    validate against JSON data, and save the cropped images and OCR results.

    Args:
        main_dir (str): Path to the main directory containing subdirectories with triplet files.
        output_dir (str): Path to the directory where cropped images and OCR results will be saved.
        ocr_output_suffix (str): Suffix for the OCR results CSV file.
        image_extension (str): Extension of the full-page image files.
        csv_extension (str): Extension of the CSV files containing bounding boxes.
        html_extension (str): Extension of the HTML files (not used but part of the triplet).
        categories_present (bool): Indicates if the CSV contains a 'category' column.
        languages (list): Languages for EasyOCR.
        use_gpu (bool): Whether to use GPU for EasyOCR. Set to False if no GPU is available.
    """
    main_path = Path(main_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Gather all collection directories
    collection_dirs = [d for d in main_path.iterdir() if d.is_dir()]
    total_collections = len(collection_dirs)
    if total_collections == 0:
        logging.error(f"No collection directories found in main directory: {main_dir}")
        return

    logging.info(f"Found {total_collections} collections to process.")

    # Shuffle the collection directories
    random.shuffle(collection_dirs)

    # Use tqdm to display progress
    for collection_dir in tqdm(collection_dirs, total=total_collections, desc="Processing Collections"):
        process_collection(
            collection_path=collection_dir,
            output_dir=output_path,
            ocr_output_suffix=ocr_output_suffix,
            image_extension=image_extension,
            csv_extension=csv_extension,
            html_extension=html_extension,
            categories_present=categories_present,
            languages=languages,
            use_gpu=use_gpu
        )

def cleanup_box_images(output_dir):
    """Clean up all box-images directories after processing"""
    for collection_dir in Path(output_dir).iterdir():
        if collection_dir.is_dir():
            box_images_dir = collection_dir / "box-images"
            if box_images_dir.exists():
                shutil.rmtree(box_images_dir)
                logging.info(f"Cleaned up box images for collection: {collection_dir.name}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Crop images based on bounding boxes from CSV files, perform OCR, validate against JSON data, and save the results.")
    parser.add_argument(
        "--main_dir",
        type=str,
        required=True,
        help="Path to the main directory containing subdirectories with triplet files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where cropped images and OCR results will be saved."
    )
    parser.add_argument(
        "--ocr_output_suffix",
        type=str,
        default="_ocr_results.csv",
        help="Suffix for the OCR results CSV file."
    )
    parser.add_argument(
        "--image_extension",
        type=str,
        default=".fullpage.jpg",
        help="Extension of the full-page image files."
    )
    parser.add_argument(
        "--csv_extension",
        type=str,
        default=".csv",
        help="Extension of the CSV files containing bounding boxes."
    )
    parser.add_argument(
        "--html_extension",
        type=str,
        default=".html",
        help="Extension of the HTML files (not used but part of the triplet)."
    )
    parser.add_argument(
        "--categories_present",
        action='store_true',
        help="Flag indicating if the CSV contains a 'category' column."
    )
    parser.add_argument(
        "--languages",
        nargs='+',
        default=['en'],
        help="List of languages for OCR. Example: --languages en fr"
    )
    parser.add_argument(
        "--use_gpu",
        action='store_true',
        help="Flag to enable GPU usage for OCR. Default is CPU."
    )

    args = parser.parse_args()

    create_cropped_images_and_perform_ocr(
        main_dir=args.main_dir,
        output_dir=args.output_dir,
        ocr_output_suffix=args.ocr_output_suffix,
        image_extension=args.image_extension,
        csv_extension=args.csv_extension,
        html_extension=args.html_extension,
        categories_present=args.categories_present,
        languages=args.languages,
        use_gpu=args.use_gpu
    )
    
    # Add cleanup at the end
    cleanup_box_images(args.output_dir)

if __name__ == "__main__":
    main()
