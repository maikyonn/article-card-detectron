"""
Bounding Box Processing Script

This script processes HTML files within a specified download directory to extract bounding boxes
using a bounding box algorithm. It leverages Playwright for browser automation and asyncio for
concurrent processing. The results are saved as CSV files in the same directory as the corresponding
HTML files. Logging is used to track the processing status and any errors encountered.

Parameters:
    --download-dir (str): Path to the directory containing HTML files to be processed.
    --max-concurrent-tasks (int): Maximum number of HTML files to process concurrently.
    --output-dir (str): Path to the directory where results will be saved.

Usage:
    python bounding_box_processor.py --download-dir "path/to/downloads" --max-concurrent-tasks 60 --output-dir "path/to/results"
"""

import os
import asyncio
import logging
import pandas as pd
from tqdm.asyncio import tqdm
from pathlib import Path
from playwright.async_api import async_playwright
import argparse
import shutil
import random
# Import your bounding box module
import box_utils as bb  # Ensure this is correctly installed or accessible

# Configure logging
logging.basicConfig(
    filename='bounding_box_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def process_html_file(semaphore, context, html_file_path, output_dir):
    """Process a single HTML file with better error handling and resource management"""
    page = None
    async with semaphore:
        try:
            # Define paths
            html_file = Path(html_file_path)
            relative_path = html_file.relative_to(args.download_dir)
            output_path = Path(output_dir) / relative_path.parent
            output_path.mkdir(parents=True, exist_ok=True)

            # Check if CSV file already exists
            result_filename = f'{html_file.stem}-new.csv'
            result_path = output_path / result_filename
            if result_path.exists():
                logging.info(f"Skipping {html_file_path} as CSV file already exists.")
                return

            # Create new page with timeout
            page = await asyncio.wait_for(context.new_page(), timeout=30)
            
            # Set longer timeouts for navigation and other operations
            page.set_default_timeout(60000)  # 60 seconds timeout
            
            try:
                browser_file_path = f"file://{html_file.resolve()}"
                await page.goto(browser_file_path, wait_until="load", timeout=60000)
                await auto_scroll(page)
                await bb.async_load_model_files_and_helper_scripts(page)
                bounding_box = await bb.get_bounding_box_one_file(page, file=browser_file_path)

                screenshot_filename = f"{html_file.stem}-new.fullpage.jpg"
                screenshot_path = output_path / screenshot_filename
                await page.screenshot(path=str(screenshot_path), full_page=True, type="jpeg", quality=90)

                bounding_box_data = bounding_box.get("bounding_boxes", {})
                bounding_box_df = pd.DataFrame(bounding_box_data)
                bounding_box_df.to_csv(result_path, index=False)

                logging.info(f"Successfully processed {html_file_path}")
                
            except Exception as e:
                logging.error(f"Error processing {html_file_path}: {str(e)}")
                raise  # Re-raise the exception to be caught by outer try-except

        except Exception as e:
            logging.error(f"Error processing {html_file_path}: {str(e)}")
            print(f"Error processing {html_file_path}: {str(e)}")
            
        finally:
            if page:
                try:
                    await page.close()
                except Exception as e:
                    logging.error(f"Error closing page for {html_file_path}: {str(e)}")

async def auto_scroll(page, total_scroll=30000, scroll_step=3000, scroll_delay=0.1):
    """
    Scrolls down the page in fixed increments until a total scroll distance is reached
    or the bottom of the page is detected.

    Args:
        page: Playwright page instance.
        total_scroll (int): Total pixels to scroll down. Defaults to 30,000.
        scroll_step (int): Pixels to scroll each step. Defaults to 3,000.
        scroll_delay (float): Delay between scrolls in seconds. Defaults to 0.1.
    """
    try:
        current_scroll = 0
        await page.evaluate("window.scrollTo(0, 0)")  # Ensure starting at the top

        while current_scroll < total_scroll:
            await page.evaluate(f"window.scrollBy(0, {scroll_step});")
            current_scroll += scroll_step
            await asyncio.sleep(scroll_delay)

            # Check if we've reached the bottom of the page
            new_height = await page.evaluate("document.body.scrollHeight")
            current_offset = await page.evaluate("window.pageYOffset + window.innerHeight")
            if current_offset >= new_height:
                logging.info("Reached the bottom of the page.")
                break

        logging.info(f"Scrolled a total of {current_scroll} pixels.")
    except Exception as e:
        logging.error(f"Auto-scroll failed: {e}")

async def main(download_dir, output_dir, max_concurrent_tasks, start_folder=None, end_folder=None):
    """Main function with folder range"""
    # Get all unique folders that contain HTML files
    html_folders = set(Path(html_file).parent for html_file in Path(download_dir).rglob('*.html'))
    html_folders = sorted(html_folders)  # Sort folders for consistent ordering
    
    # Apply folder range if specified
    if start_folder is not None or end_folder is not None:
        start_idx = start_folder if start_folder is not None else 0
        end_idx = end_folder if end_folder is not None else len(html_folders)
        html_folders = html_folders[start_idx:end_idx]
        logging.info(f"Processing folders from index {start_idx} to {end_idx}")
        print(f"Processing folders from index {start_idx} to {end_idx}")
    
    # Get all HTML files from selected folders
    html_files = []
    for folder in html_folders:
        html_files.extend(folder.glob('*.html'))
    
    if not html_files:
        logging.info(f"No HTML files found in the selected folder range.")
        return

    # Reduce max_concurrent_tasks to prevent overwhelming the browser
    max_concurrent_tasks = min(max_concurrent_tasks, 30)  # Cap at 30 concurrent tasks
    
    playwright = None
    browser = None
    context = None

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=['--disable-dev-shm-usage']  # Helps with memory issues
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            java_script_enabled=True,
            ignore_https_errors=True
        )

        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Process files in smaller batches
        batch_size = 100
        for i in range(0, len(html_files), batch_size):
            batch = html_files[i:i + batch_size]
            tasks = [
                process_html_file(semaphore, context, str(html_file), output_dir)
                for html_file in batch
            ]
            
            # Process batch with progress bar
            for f in tqdm(
                asyncio.as_completed(tasks),
                total=len(batch),
                desc=f"Processing batch {i//batch_size + 1}/{len(html_files)//batch_size + 1}"
            ):
                try:
                    await f
                except Exception as e:
                    logging.error(f"Batch processing error: {str(e)}")
                    continue

            # Short pause between batches
            await asyncio.sleep(1)

    except Exception as e:
        logging.error(f"Main process error: {str(e)}")
        print(f"Error in main process: {str(e)}")
        
    finally:
        try:
            if context:
                await context.close()
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            print(f"Error during cleanup: {str(e)}")

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
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory where results will be saved."
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=60,
        help="Maximum number of HTML files to process concurrently. Default is 60."
    )
    parser.add_argument(
        "--start-folder",
        type=int,
        default=None,
        help="Starting folder index (0-based). Default is 0."
    )
    parser.add_argument(
        "--end-folder",
        type=int,
        default=None,
        help="Ending folder index (exclusive). Default is all folders."
    )
    args = parser.parse_args()

    # Run the main function with provided arguments
    asyncio.run(main(
        args.download_dir, 
        args.output_dir, 
        args.max_concurrent_tasks,
        args.start_folder,
        args.end_folder
    ))