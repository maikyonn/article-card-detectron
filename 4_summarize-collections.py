#!/usr/bin/env python3
"""
summarize_collections.py

This script summarizes the similarity statistics of all collections by calculating the average similarity score
(weighted by average_confidence) and the percentage distribution of similarity categories (high, medium, low, very low).
It ranks the collections from best to least similar and outputs a summary text file with the results.
Additionally, it exports the names of the top N collections based on the average similarity score into a separate text file.

Usage:
    python summarize_collections.py --output_dir /path/to/output_directory --summary_file summary.txt [--top_n 500 --top_collections_file top_500_collections.txt]

Dependencies:
    - pandas
    - argparse
    - pathlib
    - tqdm
"""

import os
import pandas as pd
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

def setup_logging():
    """
    Configures the logging settings to log information to both file and console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler('summarize_collections.log')
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
    parser = argparse.ArgumentParser(description="Summarize similarity statistics of all collections and export top N collections.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="16k-dataset-5",
        help="Path to the main output directory containing subdirectories for each collection."
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="collections_summary.txt",
        help="Filename for the summary text file. *(Default: collections_summary.txt)*"
    )
    # New Argument: Number of Top Collections to Export
    parser.add_argument(
        "--top_n",
        type=int,
        default=500,
        help="Number of top collections to export based on average similarity score. *(Default: 500)*"
    )
    # New Argument: Filename for Top Collections
    parser.add_argument(
        "--top_collections_file",
        type=str,
        default="top_500_collections.txt",
        help="Filename for the top N collections text file. *(Default: top_500_collections.txt)*"
    )
    return parser.parse_args()

def summarize_collections(output_dir, summary_file, top_n, top_collections_file):
    """
    Summarizes the similarity statistics of all collections and exports the top N collection names.

    Args:
        output_dir (str): Path to the main output directory.
        summary_file (str): Path and filename for the summary text file.
        top_n (int): Number of top collections to export.
        top_collections_file (str): Filename for the top N collections text file.

    Returns:
        None
    """
    output_path = Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        logging.error(f"The specified output directory does not exist or is not a directory: {output_dir}")
        return

    # Gather all collection directories
    collection_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    total_collections = len(collection_dirs)

    if total_collections == 0:
        logging.error(f"No collection directories found in output directory: {output_dir}")
        return

    logging.info(f"Found {total_collections} collections to summarize.")

    # Initialize list to hold summary data
    summary_data = []

    # Iterate through each collection directory with progress bar
    for collection_dir in tqdm(collection_dirs, desc="Summarizing Collections"):
        collection_name = collection_dir.name
        ocr_csv_files = list(collection_dir.glob("*_ocr_results.csv"))

        if not ocr_csv_files:
            logging.warning(f"No OCR results CSV found in collection: {collection_name}. Skipping.")
            continue

        # Initialize a DataFrame to accumulate OCR results
        collection_ocr_df = pd.DataFrame()

        for ocr_csv in ocr_csv_files:
            try:
                df = pd.read_csv(ocr_csv)
                collection_ocr_df = pd.concat([collection_ocr_df, df], ignore_index=True)
                logging.info(f"Read OCR results from {ocr_csv} for collection {collection_name}.")
            except Exception as e:
                logging.error(f"Failed to read OCR CSV file {ocr_csv} in collection {collection_name}: {e}")
                continue

        if collection_ocr_df.empty:
            logging.warning(f"No valid OCR data found for collection {collection_name}. Skipping.")
            continue

        # Check if 'similarity_score' and 'average_confidence' columns exist
        if 'similarity_score' not in collection_ocr_df.columns:
            logging.error(f"'similarity_score' column missing in OCR CSV for collection {collection_name}. Skipping.")
            continue
        if 'average_confidence' not in collection_ocr_df.columns:
            logging.error(f"'average_confidence' column missing in OCR CSV for collection {collection_name}. Skipping.")
            continue

        # Ensure that 'average_confidence' values are numeric and between 0 and 1
        try:
            collection_ocr_df['average_confidence'] = pd.to_numeric(collection_ocr_df['average_confidence'], errors='coerce')
            if collection_ocr_df['average_confidence'].isnull().any():
                raise ValueError("Non-numeric average_confidence values found.")
            if not ((collection_ocr_df['average_confidence'] >= 0) & (collection_ocr_df['average_confidence'] <= 1)).all():
                raise ValueError("average_confidence values must be between 0 and 1.")
        except Exception as e:
            logging.error(f"Invalid 'average_confidence' values in collection {collection_name}: {e}. Skipping.")
            continue

        # Calculate weighted average similarity
        total_confidence = collection_ocr_df['average_confidence'].sum()
        if total_confidence == 0:
            logging.warning(f"Total average_confidence is 0 for collection {collection_name}. Skipping.")
            continue
        weighted_sum_similarity = (collection_ocr_df['similarity_score'] * collection_ocr_df['average_confidence']).sum()
        weighted_average_similarity = weighted_sum_similarity / total_confidence

        # Calculate weighted counts for each similarity category
        high_similarity_weight = (collection_ocr_df['similarity_score'] >= 0.90) * collection_ocr_df['average_confidence']
        medium_similarity_weight = ((collection_ocr_df['similarity_score'] >= 0.70) & (collection_ocr_df['similarity_score'] < 0.90)) * collection_ocr_df['average_confidence']
        low_similarity_weight = ((collection_ocr_df['similarity_score'] >= 0.50) & (collection_ocr_df['similarity_score'] < 0.70)) * collection_ocr_df['average_confidence']
        very_low_similarity_weight = (collection_ocr_df['similarity_score'] < 0.50) * collection_ocr_df['average_confidence']

        high_similarity_count = high_similarity_weight.sum()
        medium_similarity_count = medium_similarity_weight.sum()
        low_similarity_count = low_similarity_weight.sum()
        very_low_similarity_count = very_low_similarity_weight.sum()

        # Calculate percentages for each similarity category
        high_similarity_percent = (high_similarity_count / total_confidence) * 100
        medium_similarity_percent = (medium_similarity_count / total_confidence) * 100
        low_similarity_percent = (low_similarity_count / total_confidence) * 100
        very_low_similarity_percent = (very_low_similarity_count / total_confidence) * 100

        # Append to summary data
        summary_data.append({
            'collection_name': collection_name,
            'average_similarity': weighted_average_similarity,
            'total_confidence': total_confidence,
            'high_similarity_weight': high_similarity_count,
            'high_similarity_percent': high_similarity_percent,
            'medium_similarity_weight': medium_similarity_count,
            'medium_similarity_percent': medium_similarity_percent,
            'low_similarity_weight': low_similarity_count,
            'low_similarity_percent': low_similarity_percent,
            'very_low_similarity_weight': very_low_similarity_count,
            'very_low_similarity_percent': very_low_similarity_percent
        })

    if not summary_data:
        logging.error("No summary data was collected. Please check the OCR results.")
        return

    # Create a DataFrame from summary data
    summary_df = pd.DataFrame(summary_data)

    # Sort the DataFrame from highest to lowest average similarity
    summary_df.sort_values(by='average_similarity', ascending=False, inplace=True)

    # Reset index after sorting
    summary_df.reset_index(drop=True, inplace=True)

    # Prepare summary text
    summary_lines = []
    summary_lines.append("Collections Similarity Summary (Weighted by Average Confidence)\n")
    summary_lines.append("="*150 + "\n")
    summary_lines.append(f"Total Collections Summarized: {len(summary_df)}\n\n")
    summary_lines.append("{:<30} {:<20} {:<18} {:<30} {:<30} {:<30} {:<30}\n".format(
        "Collection Name",
        "Avg Similarity",
        "Total Confidence",
        "High (>=0.90) Weight (%)",
        "Medium (0.70-0.89) Weight (%)",
        "Low (0.50-0.69) Weight (%)",
        "Very Low (<0.50) Weight (%)"
    ))
    summary_lines.append("-"*180 + "\n")

    for _, row in summary_df.iterrows():
        summary_lines.append("{:<30} {:<20.4f} {:<18.4f} {:<30.2f} {:<30.2f} {:<30.2f} {:<30.2f}\n".format(
            row['collection_name'],
            row['average_similarity'],
            row['total_confidence'],
            row['high_similarity_percent'],
            row['medium_similarity_percent'],
            row['low_similarity_percent'],
            row['very_low_similarity_percent']
        ))

    # Write summary to the text file
    try:
        with open(summary_file, 'w') as f:
            f.writelines(summary_lines)
        logging.info(f"Successfully wrote summary to {summary_file}")
    except Exception as e:
        logging.error(f"Failed to write summary to {summary_file}: {e}")

    # New Functionality: Export Top N Collection Names to a Text File
    try:
        # Determine the actual number of top collections (in case there are fewer than top_n)
        actual_top_n = min(top_n, len(summary_df))
        top_collections = summary_df['collection_name'].head(actual_top_n).tolist()

        # Write the top collection names to the specified text file
        with open(top_collections_file, 'w') as f:
            for collection_name in top_collections:
                f.write(f"{collection_name}\n")
        logging.info(f"Successfully wrote top {actual_top_n} collection names to {top_collections_file}")
    except Exception as e:
        logging.error(f"Failed to write top {top_n} collections to {top_collections_file}: {e}")

def main():
    setup_logging()
    args = parse_arguments()
    summarize_collections(args.output_dir, args.summary_file, args.top_n, args.top_collections_file)

if __name__ == "__main__":
    main()