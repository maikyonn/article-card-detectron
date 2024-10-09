#!/usr/bin/env python3
"""
prune_csvs.py

This script prunes OCR result CSV files based on similarity scores. It adds 'matched_text' and 'weighted_similarity_score'
columns to each CSV if not already present and creates a pruned version containing only entries with good similarity scores.

Usage:
    python prune_csvs.py 
        --input_dir /path/to/original_csvs 
        --output_dir /path/to/pruned_csvs 
        --prune_method threshold 
        --similarity_threshold 0.8
    or
    python prune_csvs.py 
        --input_dir /path/to/original_csvs 
        --output_dir /path/to/pruned_csvs 
        --prune_method percentile 
        --percentile 80

Arguments:
    --input_dir: Directory containing the original OCR CSV files.
    --output_dir: Directory where pruned CSV files will be saved.
    --prune_method: Method to define "good" scores ('threshold' or 'percentile').
    --similarity_threshold: (If prune_method is 'threshold') The minimum similarity score to retain.
    --percentile: (If prune_method is 'percentile') The percentile threshold to retain top entries.

Dependencies:
    - pandas
    - argparse
    - pathlib
    - logging
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
    fh = logging.FileHandler('prune_csvs.log')
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
    parser = argparse.ArgumentParser(description="Prune OCR result CSV files based on similarity scores.")

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to the directory containing original OCR CSV files.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the directory where pruned CSV files will be saved.'
    )
    parser.add_argument(
        '--prune_method',
        type=str,
        choices=['threshold', 'percentile'],
        required=True,
        help="Method to define 'good' scores: 'threshold' or 'percentile'."
    )
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=0.8,
        help="(If prune_method is 'threshold') The minimum similarity score to retain. *(Default: 0.8)*"
    )
    parser.add_argument(
        '--percentile',
        type=float,
        default=80.0,
        help="(If prune_method is 'percentile') The percentile threshold to retain top entries. *(Default: 80.0)*"
    )

    return parser.parse_args()

def add_columns_if_missing(df):
    """
    Adds 'matched_text' and 'weighted_similarity_score' columns to the DataFrame if they are missing.

    Args:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: DataFrame with necessary columns.
    """
    if 'matched_text' not in df.columns:
        df['matched_text'] = ""
        logging.info("Added missing 'matched_text' column.")
    if 'weighted_similarity_score' not in df.columns:
        df['weighted_similarity_score'] = 0.0
        logging.info("Added missing 'weighted_similarity_score' column.")
    return df

def prune_df(df, method, threshold=None, percentile=None):
    """
    Prunes the DataFrame based on the specified method.

    Args:
        df (pd.DataFrame): DataFrame to prune.
        method (str): 'threshold' or 'percentile'.
        threshold (float, optional): Similarity score threshold.
        percentile (float, optional): Percentile for pruning.

    Returns:
        pd.DataFrame: Pruned DataFrame.
    """
    if method == 'threshold':
        pruned_df = df[df['weighted_similarity_score'] >= threshold].copy()
        logging.info(f"Pruned DataFrame using threshold ≥ {threshold}. Retained {len(pruned_df)} out of {len(df)} entries.")
    elif method == 'percentile':
        cutoff = df['weighted_similarity_score'].quantile(percentile / 100)
        pruned_df = df[df['weighted_similarity_score'] >= cutoff].copy()
        logging.info(f"Pruned DataFrame using top {percentile}% percentile (cutoff: {cutoff:.4f}). Retained {len(pruned_df)} out of {len(df)} entries.")
    else:
        logging.error(f"Unknown prune method: {method}. No pruning applied.")
        pruned_df = df.copy()
    return pruned_df

def process_csv_file(csv_path, output_path, prune_method, similarity_threshold, percentile):
    """
    Processes a single CSV file: adds missing columns, prunes based on similarity score, and saves the pruned CSV.

    Args:
        csv_path (Path): Path to the original CSV file.
        output_path (Path): Path to save the pruned CSV file.
        prune_method (str): 'threshold' or 'percentile'.
        similarity_threshold (float): Threshold value for 'threshold' method.
        percentile (float): Percentile value for 'percentile' method.

    Returns:
        None
    """
    try:
        df = pd.read_csv(csv_path)
        df = add_columns_if_missing(df)

        # Check if 'weighted_similarity_score' column exists and has valid values
        if 'weighted_similarity_score' not in df.columns:
            logging.warning(f"'weighted_similarity_score' column missing in '{csv_path.name}'. All entries set to 0.0.")
            df['weighted_similarity_score'] = 0.0
        else:
            # Ensure 'weighted_similarity_score' is numeric
            df['weighted_similarity_score'] = pd.to_numeric(df['weighted_similarity_score'], errors='coerce').fillna(0.0)

        # Prune the DataFrame
        pruned_df = prune_df(df, prune_method, threshold=similarity_threshold, percentile=percentile)

        # Save the pruned DataFrame
        pruned_df.to_csv(output_path, index=False)
        logging.info(f"Saved pruned CSV to '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to process CSV file '{csv_path}': {e}")

def main():
    setup_logging()
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prune_method = args.prune_method
    similarity_threshold = args.similarity_threshold
    percentile = args.percentile

    # Gather all CSV files in the input directory
    csv_files = list(input_dir.glob('*.csv'))
    total_files = len(csv_files)
    if total_files == 0:
        logging.error(f"No CSV files found in the input directory: {input_dir}")
        return

    logging.info(f"Found {total_files} CSV file(s) to process.")

    # Process each CSV file with a progress bar
    for csv_file in tqdm(csv_files, desc="Pruning CSVs"):
        relative_path = csv_file.relative_to(input_dir)
        pruned_csv_path = output_dir / relative_path

        # Ensure the output subdirectory exists
        pruned_csv_path.parent.mkdir(parents=True, exist_ok=True)

        process_csv_file(
            csv_path=csv_file,
            output_path=pruned_csv_path,
            prune_method=prune_method,
            similarity_threshold=similarity_threshold,
            percentile=percentile
        )

    logging.info("Pruning of all CSV files is complete.")

if __name__ == "__main__":
    main()