"""
Bounding Box Processing Script

This script processes HTML files within a specified download directory to extract bounding boxes
using a bounding box algorithm. It leverages Playwright for browser automation and asyncio for
concurrent processing. The results are saved as CSV files in the same directory as the corresponding
HTML files. Logging is used to track the processing status and any errors encountered.

Parameters:
    --download-dir (str): Path to the directory containing HTML files to be processed.
    --max-concurrent-tasks (int): Maximum number of HTML files to process concurrently.

Usage:
    python bounding_box_processor.py --download-dir "path/to/downloads" --max-concurrent-tasks 60
"""

import os
import asyncio
import logging
import pandas as pd
from tqdm.asyncio import tqdm
from pathlib import Path
from playwright.async_api import async_playwright
import argparse

# Import your bounding box module
import box_utils as bb  # Ensure this is correctly installed or accessible

# Configure logging
logging.basicConfig(
    filename='bounding_box_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def process_html_file(semaphore, context, html_file_path):
    """
    Process a single HTML file to extract bounding boxes and save results as a CSV
    in the same directory as the HTML file.

    Args:
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.
        context (playwright.async_api.BrowserContext): Playwright browser context.
        html_file_path (str): Path to the HTML file.
    """
    async with semaphore:
        page = await context.new_page()
        try:
            # Define paths
            html_file = Path(html_file_path)
            download_path = html_file.resolve()
            browser_fp = f'file://{download_path}'

            # Navigate to the HTML file in the browser
            await page.goto(browser_fp, wait_until="load")
            # Apply bounding box algorithm
            await bb.async_load_model_files_and_helper_scripts(page)
            bounding_box = await bb.get_bounding_box_one_file(page, file=browser_fp)
            await bb.draw_visual_bounding_boxes_on_page(page, file=browser_fp)

            # Prepare the result path in the same directory as the HTML file
            result_filename = f'{html_file.stem}.csv'
            result_path = html_file.parent / result_filename

            # Convert bounding boxes to DataFrame and save as CSV
            bounding_box_data = bounding_box.get("bounding_boxes", {})

            # Ensure bounding_box_data is in a suitable format for DataFrame
            # Adjust the following line based on the actual structure of bounding_box_data
            bounding_box_df = pd.DataFrame(bounding_box_data)

            bounding_box_df.to_csv(result_path, index=False)

            logging.info(f"Successfully processed {html_file_path} and saved results to {result_path}")
            print(f"Processed: {html_file_path}")

        except Exception as e:
            logging.error(f"Error processing {html_file_path}: {e}")
            print(f"Error processing {html_file_path}: {e}")
        finally:
            await page.close()

async def main(download_dir, max_concurrent_tasks):
    """
    Main function to process all HTML files in the specified downloads directory.
    Initializes Playwright, manages concurrency, and processes files.

    Args:
        download_dir (str): Path to the directory containing HTML files to be processed.
        max_concurrent_tasks (int): Maximum number of HTML files to process concurrently.
    """
    # Gather all HTML files recursively in the downloads directory
    html_files = list(Path(download_dir).rglob('*.html'))

    if not html_files:
        logging.info(f"No HTML files found in {download_dir}.")
        print(f"No HTML files found in {download_dir}.")
        return

    # Initialize Playwright and launch browser
    async with async_playwright() as p:
        # Launch the browser (you can choose 'chromium', 'firefox', or 'webkit')
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        
        # Initialize semaphore for limiting concurrency
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Create tasks for processing HTML files
        tasks = [
            process_html_file(semaphore, context, str(html_file))
            for html_file in html_files
        ]

        # Execute tasks with progress bar
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing HTML files"):
            await f

        # Close browser
        await browser.close()

    logging.info("All HTML files have been processed.")
    print("All HTML files have been processed.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process HTML files to extract bounding boxes.")
    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Path to the directory containing HTML files to be processed."
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=60,
        help="Maximum number of HTML files to process concurrently. Default is 60."
    )
    args = parser.parse_args()

    # Run the main function with provided arguments
    asyncio.run(main(args.download_dir, args.max_concurrent_tasks))