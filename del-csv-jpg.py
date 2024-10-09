#!/usr/bin/env python3
"""
Multithreaded File Deletion Script

This script deletes .csv files and .jpg files (excluding .fullpage.jpg) in a specified directory.
It utilizes multithreading to perform deletions concurrently for improved performance.
It also removes empty folders after file deletion.

Usage:
    python delete_files_multithreaded.py --directory "/path/to/directory" [--recursive] [--dry-run]

Arguments:
    --directory (-d): Path to the target directory.
    --recursive (-r): (Optional) Include subdirectories in the deletion process.
    --dry-run (-n): (Optional) Perform a trial run without deleting any files or folders.
"""

import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import sys

def configure_logging(log_file: Path):
    """Configures logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def find_files(target_dir: Path, extensions: list, recursive: bool) -> list:
    """
    Finds all files in target_dir with the given extensions, excluding .fullpage.jpg files.

    Args:
        target_dir (Path): Directory to search.
        extensions (list): List of file extensions to include.
        recursive (bool): Whether to search subdirectories.

    Returns:
        list: List of file paths matching the criteria.
    """
    if recursive:
        all_files = target_dir.rglob('*')
    else:
        all_files = target_dir.glob('*')
    
    return [f for f in all_files if f.is_file() and 
            (f.suffix.lower() in extensions and not f.name.lower().endswith('.fullpage.jpg'))]

def delete_file(file_path: Path, dry_run: bool=False) -> tuple:
    """
    Deletes the specified file.

    Args:
        file_path (Path): Path to the file to delete.
        dry_run (bool): If True, does not delete the file.

    Returns:
        tuple: (file_path, success:bool, message:str)
    """
    try:
        if dry_run:
            return (file_path, True, "Dry run: not deleted.")
        file_path.unlink()
        return (file_path, True, "Deleted successfully.")
    except Exception as e:
        return (file_path, False, f"Error: {e}")

def remove_empty_folders(directory: Path, dry_run: bool=False) -> None:
    """
    Removes empty folders in the given directory.

    Args:
        directory (Path): Directory to clean up.
        dry_run (bool): If True, does not delete any folders.
    """
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        if not dirnames and not filenames:
            try:
                if not dry_run:
                    os.rmdir(dirpath)
                    logging.info(f"Removed empty folder: {dirpath}")
                else:
                    logging.info(f"Dry run: Would remove empty folder: {dirpath}")
            except Exception as e:
                logging.error(f"Error removing empty folder {dirpath}: {e}")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Multithreaded script to delete .csv and .jpg files (excluding .fullpage.jpg) in a directory and remove empty folders.")
    parser.add_argument(
        '--directory', '-d',
        type=str,
        required=True,
        help='Path to the target directory.'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Include subdirectories in the deletion process.'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Perform a trial run without deleting any files or folders.'
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    target_dir = Path(args.directory).resolve()

    # Validate directory
    if not target_dir.is_dir():
        print(f"Error: The specified directory does not exist or is not a directory: {target_dir}")
        sys.exit(1)

    # Configure logging
    log_file = target_dir / 'deletion.log'
    configure_logging(log_file)

    # Confirmation prompt (unless dry run)
    if not args.dry_run:
        confirm = input(f"WARNING: This will delete all .csv and .jpg files (excluding .fullpage.jpg) in {target_dir} and remove empty folders.\nDo you want to continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled by user.")
            sys.exit(0)
    else:
        print("Performing a dry run. No files or folders will be deleted.")

    # Find files to delete
    extensions = ['.csv', '.jpg']
    files_to_delete = find_files(target_dir, extensions, args.recursive)

    if not files_to_delete:
        logging.info("No .csv or .jpg files (excluding .fullpage.jpg) found to delete.")
        print("No .csv or .jpg files (excluding .fullpage.jpg) found to delete.")
    else:
        logging.info(f"Found {len(files_to_delete)} files to delete.")
        print(f"Found {len(files_to_delete)} files to delete.")

        # Set up multithreading
        max_workers = min(32, os.cpu_count() + 4)  # Adjust number of threads as needed

        # Initialize progress bar
        progress_bar = tqdm(total=len(files_to_delete), desc="Deleting files", unit="file")

        # Use ThreadPoolExecutor to delete files concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all deletion tasks
            future_to_file = {executor.submit(delete_file, file_path, args.dry_run): file_path for file_path in files_to_delete}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file, success, message = future.result()
                    if success:
                        logging.info(f"{file}: {message}")
                    else:
                        logging.error(f"{file}: {message}")
                except Exception as exc:
                    logging.error(f"{file_path}: Generated an exception: {exc}")
                finally:
                    progress_bar.update(1)

        progress_bar.close()
        logging.info("File deletion process completed.")
        print("File deletion process completed.")

    # Remove empty folders
    logging.info("Starting to remove empty folders.")
    print("Removing empty folders...")
    remove_empty_folders(target_dir, args.dry_run)
    logging.info("Empty folder removal process completed.")
    print("Empty folder removal process completed.")

    print(f"Detailed logs can be found in {log_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)