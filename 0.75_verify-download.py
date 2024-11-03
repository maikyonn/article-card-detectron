"""
A script to validate and clean downloaded files from Internet Archive collections.

This script checks folders containing downloaded files from Internet Archive collections,
verifying that each folder has the expected number of HTML and JPG files. It can also
validate the integrity of HTML and JPG files, and optionally remove invalid folders.

Basic check for correct number of files:
    python 0.75_verify-download.py --downloads-dir 16k-dataset --required-files 15

Check files and remove invalid folders:
    python 0.75_verify-download.py \
        --downloads-dir 16k-dataset \
        --required-files 15 \
        --remove-invalid

Full validation with HTML and image checks:
    python 0.75_verify-download.py \
        --downloads-dir 16k-dataset \
        --required-files 15 \
        --remove-invalid \
        --validate-html \
        --validate-images

Arguments:
    --downloads-dir: Path to directory containing downloaded folders (default: 16k-dataset)
    --required-files: Number of HTML and JPG files each folder should have (default: 15)
    --remove-invalid: Remove folders that fail validation checks
    --validate-html: Enable HTML file validation
    --validate-images: Enable JPG file validation
"""

import os
import shutil
import argparse
from PIL import Image, UnidentifiedImageError
from bs4 import BeautifulSoup
from tqdm import tqdm

def validate_html(file_path):
    """
    Validates an HTML file by attempting to parse it with BeautifulSoup.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
        return True
    except Exception as e:
        print(f"HTML Validation Error in '{file_path}': {e}")
        return False

def validate_jpg(file_path):
    """
    Validates a JPG file by attempting to open it with Pillow.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it's an image
        return True
    except UnidentifiedImageError:
        print(f"JPG Validation Error: '{file_path}' is not a valid image.")
        return False
    except Exception as e:
        print(f"JPG Validation Error in '{file_path}': {e}")
        return False

def check_and_clean_folders(downloads_dir, required_files=3, remove_invalid=True, validate_html_flag=False, validate_images_flag=False):
    """
    Iterates through each folder in the downloads directory, checks for the required files,
    validates them based on provided flags, and optionally removes the folder if any check fails.
    """
    # Get list of all subdirectories in downloads_dir
    subfolders = [os.path.join(downloads_dir, d) for d in os.listdir(downloads_dir)
                  if os.path.isdir(os.path.join(downloads_dir, d))]

    bad_directories = 0
    bad_html_files = 0
    bad_jpg_files = 0
    total_valid_folders = 0
    total_folders = len(subfolders)
    bad_directories_list = []  # List to store tuples of bad directories and their file counts

    # Initialize progress bar
    with tqdm(total=total_folders, desc="Processing folders") as pbar:
        for folder in subfolders:
            folder_name = os.path.basename(folder)
            print(f"\nProcessing folder: {folder_name}")

            # List all files in the folder
            all_files = os.listdir(folder)
            total_files = len(all_files)

            # Filter .html and .jpg files
            html_files = [f for f in all_files if f.lower().endswith('.html')]
            jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]

            # Check if there are exactly required_files .html and .jpg files
            if len(html_files) != required_files or len(jpg_files) != required_files:
                print(f"Incorrect number of files in '{folder_name}'. Expected {required_files} .html and {required_files} .jpg files, found {len(html_files)} .html and {len(jpg_files)} .jpg files.")
                if remove_invalid:
                    remove_folder(folder)
                bad_directories += 1
                bad_directories_list.append((folder_name, total_files))  # Add to bad directories list with file count
                pbar.update(1)
                continue

            # Initialize flags to track validation
            html_valid = True
            jpg_valid = True

            # Validate HTML files if the flag is set
            if validate_html_flag:
                for html in html_files:
                    html_path = os.path.join(folder, html)
                    if not validate_html(html_path):
                        html_valid = False
                        bad_html_files += 1
                        break  # No need to check further if one fails
                if not html_valid:
                    print(f"Invalid HTML files detected in '{folder_name}'.")
                    if remove_invalid:
                        print("Removing folder.")
                        remove_folder(folder)
                    bad_directories += 1
                    bad_directories_list.append((folder_name, total_files))  # Add to bad directories list with file count
                    pbar.update(1)
                    continue

            # Validate JPG files if the flag is set
            if validate_images_flag:
                for jpg in jpg_files:
                    jpg_path = os.path.join(folder, jpg)
                    if not validate_jpg(jpg_path):
                        jpg_valid = False
                        bad_jpg_files += 1
                        break  # No need to check further if one fails
                if not jpg_valid:
                    print(f"Invalid JPG files detected in '{folder_name}'.")
                    if remove_invalid:
                        print("Removing folder.")
                        remove_folder(folder)
                    bad_directories += 1
                    bad_directories_list.append((folder_name, total_files))  # Add to bad directories list with file count
                    pbar.update(1)
                    continue

            # If all checks pass or validations are skipped
            print(f"Folder '{folder_name}' is valid.")
            total_valid_folders += 1
            pbar.update(1)

    print("\nValidation and cleanup complete!")
    print(f"Summary:")
    print(f"- Total folders processed: {total_folders}")
    print(f"- Valid folders: {total_valid_folders}")
    print(f"- Bad directories removed: {bad_directories}")
    if validate_html_flag:
        print(f"- Bad HTML files found: {bad_html_files}")
    if validate_images_flag:
        print(f"- Bad JPG files found: {bad_jpg_files}")
    
    # Display the list of bad directories with their file counts
    if bad_directories_list:
        print("\nList of Bad Directories:")
        for idx, (dir_name, file_count) in enumerate(bad_directories_list, 1):
            print(f"{idx}. {dir_name} - {file_count} files")
    else:
        print("\nNo bad directories found.")

def remove_folder(folder_path):
    """
    Removes the specified folder and all its contents.
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Removed folder: {folder_path}")
    except Exception as e:
        print(f"Error removing folder '{folder_path}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Validate and clean download folders.")
    parser.add_argument('--downloads-dir', type=str, default='16k-dataset',
                        help="Path to the downloads directory. Default is 'downloads-16k-proxy'.")
    parser.add_argument('--required-files', type=int, default=15,  # Changed from 15 to 3 to match docstring
                        help="Number of required HTML and JPG files in each folder. Default is 3.")
    parser.add_argument('--remove-invalid', action='store_true',
                        help="Remove invalid folders. If not set, the script will only report issues without removing folders.")
    parser.add_argument('--validate-html', action='store_true',
                        help="Enable validation of HTML files using BeautifulSoup.")
    parser.add_argument('--validate-images', action='store_true',
                        help="Enable validation of JPG images using Pillow.")
    args = parser.parse_args()

    downloads_dir = args.downloads_dir
    required_files = args.required_files
    remove_invalid = args.remove_invalid
    validate_html_flag = args.validate_html
    validate_images_flag = args.validate_images

    # Check if downloads_dir exists
    if not os.path.isdir(downloads_dir):
        print(f"Error: The directory '{downloads_dir}' does not exist.")
        return

    # Start processing folders
    check_and_clean_folders(
        downloads_dir,
        required_files=required_files,
        remove_invalid=remove_invalid,
        validate_html_flag=validate_html_flag,
        validate_images_flag=validate_images_flag
    )

if __name__ == "__main__":
    main()