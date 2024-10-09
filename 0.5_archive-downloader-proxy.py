import os
import argparse
import internetarchive as ia
from tqdm import tqdm
import logging
import time
from requests.exceptions import Timeout, ConnectionError, RequestException

# ===========================
# Configuration Section
# ===========================

class Config:
    # Logging Configuration
    LOG_FILE = 'download_log.log'
    LOG_LEVEL = logging.INFO

    # Retry Mechanism
    MAX_RETRIES = 5
    RETRY_DELAY = 5  # seconds

    # Download Configuration
    BASE_DIR = 'downloads-16k-proxy'
    COLLECTION_NAME = 'news-homepages'

    # File Types and Limits
    FILE_TYPE_LIMITS = {
        '.html': 15,
        'fullpage.jpg': 15
    }

    # Timeout Settings
    DOWNLOAD_TIMEOUT = 30  # seconds

# ===========================
# Logging Setup
# ===========================

logging.basicConfig(
    filename=Config.LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=Config.LOG_LEVEL
)

# ===========================
# Utility Functions
# ===========================

def create_directory_if_needed(identifier, base_dir=Config.BASE_DIR):
    """
    Create a directory for the given identifier if it doesn't exist.
    """
    item_dir = os.path.join(base_dir, identifier)
    os.makedirs(item_dir, exist_ok=True)
    return item_dir

def retry_operation(func, *args, **kwargs):
    """
    Retry a function with exponential backoff.
    """
    retries = 0
    while retries < Config.MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except (Timeout, ConnectionError, RequestException) as e:
            retries += 1
            logging.warning(f"Attempt {retries}/{Config.MAX_RETRIES} failed with error: {e}. Retrying in {Config.RETRY_DELAY} seconds...")
            time.sleep(Config.RETRY_DELAY)
        except Exception as e:
            logging.error(f"Unexpected error: {e}. No retry.")
            break
    logging.error(f"Operation failed after {Config.MAX_RETRIES} retries.")
    return None

# ===========================
# Core Functionality
# ===========================

def download_collection_chunk(start, end, proxy=None):
    """
    Download a chunk of items from the specified collection using the given proxy.
    """
    try:
        collection_subset = []

        # Configure proxy if provided
        if proxy:
            proxies = {
                'http': proxy,
                'https': proxy
            }
            ia.session.proxies = proxies
            logging.info(f"Using proxy: {proxy}")
            print(f"Using proxy: {proxy}")
        else:
            logging.info("No proxy configured.")
            print("No proxy configured.")

        # Perform the search and fetch data from API
        search_query = f'collection:{Config.COLLECTION_NAME}'
        search = ia.search_items(search_query)

        # Fetch search results
        logging.info("Fetching search results from Internet Archive...")
        with tqdm(desc="Fetching collection items", unit="item") as pbar:
            for result in search:
                collection_subset.append(result)
                pbar.update(1)
                if end is not None and len(collection_subset) >= end:
                    break  # Stop if we've reached the desired end

        # Adjust the subset based on start and end indices
        if end is not None:
            collection_subset = collection_subset[start:end]
        else:
            collection_subset = collection_subset[start:]

        total_items = len(collection_subset)

        if total_items == 0:
            logging.info("No items found in the specified range.")
            print("No items found in the specified range.")
            return

        # Initialize progress bar for downloading items
        with tqdm(total=total_items, desc="Processing items from collection") as pbar:
            for result in collection_subset:
                identifier = result.get('identifier', 'unknown_identifier')
                logging.info(f"Processing item: {identifier}")
                print(f"\nProcessing item: {identifier}")

                # Fetch item details with retry
                item = retry_operation(ia.get_item, identifier)
                if item is None:
                    logging.error(f"Failed to fetch item details for {identifier}. Skipping.")
                    pbar.update(1)
                    continue

                # List files and filter based on configured types and limits
                try:
                    files = list(item.get_files())
                except Exception as e:
                    logging.error(f"Error fetching files for {identifier}: {e}. Skipping.")
                    print(f"Error fetching files for {identifier}: {e}. Skipping.")
                    pbar.update(1)
                    continue

                # Counters for limiting files
                file_counts = {ft: 0 for ft in Config.FILE_TYPE_LIMITS}
                filtered_files = []

                # Process files and gather up to the specified limit for each type
                for file in files:
                    for file_type, limit in Config.FILE_TYPE_LIMITS.items():
                        if file.name.endswith(file_type) and file_counts[file_type] < limit:
                            filtered_files.append(file)
                            file_counts[file_type] += 1
                    # Check if all limits are reached
                    if all(file_counts[ft] >= limit for ft, limit in Config.FILE_TYPE_LIMITS.items()):
                        break

                # Only proceed if required file types are found
                if all(count > 0 for count in file_counts.values()):
                    logging.info(f"Downloading {len(filtered_files)} files for {identifier}.")
                    print(f"Downloading {len(filtered_files)} files for {identifier}.")

                    # Create directory only if there are files to download
                    item_dir = create_directory_if_needed(identifier)

                    # Download the files with a progress bar for the files
                    with tqdm(total=len(filtered_files), desc=f"Downloading files for {identifier}", leave=False) as file_pbar:
                        for file in filtered_files:
                            # Define the full path where the file should be saved
                            file_path = os.path.join(item_dir, file.name)

                            # Check if the file already exists and skip if it does
                            if os.path.exists(file_path):
                                logging.info(f"File {file.name} already exists. Skipping.")
                                file_pbar.update(1)
                                continue

                            # Live feed of what file is being downloaded
                            logging.info(f"Downloading file: {file.name}")
                            print(f"Downloading file: {file.name}")

                            # Download the file with retry
                            success = False
                            for attempt in range(1, Config.MAX_RETRIES + 1):
                                try:
                                    # Set a timeout for the download operation if supported
                                    file.download(destdir=item_dir, timeout=Config.DOWNLOAD_TIMEOUT)
                                    success = True
                                    break
                                except (Timeout, ConnectionError, RequestException) as e:
                                    logging.warning(f"Attempt {attempt}/{Config.MAX_RETRIES} failed to download {file.name}: {e}. Retrying in {Config.RETRY_DELAY} seconds...")
                                    time.sleep(Config.RETRY_DELAY)
                                except Exception as e:
                                    logging.error(f"Unexpected error while downloading {file.name}: {e}. No retry.")
                                    break
                            if not success:
                                logging.error(f"Failed to download {file.name} after {Config.MAX_RETRIES} attempts. Skipping.")
                                print(f"Failed to download {file.name} after {Config.MAX_RETRIES} attempts. Skipping.")

                            # Update the file progress bar
                            file_pbar.update(1)

                    # ===========================
                    # Verification Step
                    # ===========================
                    logging.info(f"Verifying downloaded files for {identifier}.")
                    print(f"Verifying downloaded files for {identifier}.")

                    try:
                        actual_files = os.listdir(item_dir)
                        verification_passed = True
                        for file_type, limit in Config.FILE_TYPE_LIMITS.items():
                            actual_count = len([f for f in actual_files if f.endswith(file_type)])
                            expected_count = Config.FILE_TYPE_LIMITS[file_type]
                            if actual_count != expected_count:
                                logging.warning(f"Verification failed for {identifier}: Expected {expected_count} '{file_type}' files, found {actual_count}.")
                                print(f"Verification failed for {identifier}: Expected {expected_count} '{file_type}' files, found {actual_count}.")
                                verification_passed = False
                        if verification_passed:
                            logging.info(f"All file type counts are correct for {identifier}.")
                            print(f"All file type counts are correct for {identifier}.")
                    except Exception as e:
                        logging.error(f"Error during verification for {identifier}: {e}")
                        print(f"Error during verification for {identifier}: {e}")

                else:
                    logging.info(f"Required file types not found together for {identifier}. Skipping.")
                    print(f"Required file types not found together for {identifier}. Skipping.")

                # Update the item progress bar
                pbar.update(1)

        logging.info("All items processed!")
        print("All items processed!")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

# ===========================
# Main Execution
# ===========================

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Download a chunk of a collection from Internet Archive through a proxy.")
    parser.add_argument('--start', type=int, default=0, help='Start index of the collection chunk (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index of the collection chunk (exclusive)')

    # Optional: Allow overriding configuration via command-line arguments
    parser.add_argument('--max_retries', type=int, help='Maximum number of retries for operations')
    parser.add_argument('--retry_delay', type=int, help='Delay between retries in seconds')
    parser.add_argument('--base_dir', type=str, help='Base directory for downloads')
    parser.add_argument('--collection', type=str, help='Name of the collection to download from')
    parser.add_argument('--file_type_limits', type=str, nargs='*', help='File type limits in the format type=limit, e.g., .html=5 fullpage.jpg=2')

    # Proxy-related arguments
    parser.add_argument('--proxy', type=str, help='Proxy server in the format socks5://username:password@IP:Port')

    args = parser.parse_args()

    start = args.start
    end = args.end

    # Override configurations if provided
    if args.max_retries is not None:
        Config.MAX_RETRIES = args.max_retries
    if args.retry_delay is not None:
        Config.RETRY_DELAY = args.retry_delay
    if args.base_dir is not None:
        Config.BASE_DIR = args.base_dir
    if args.collection is not None:
        Config.COLLECTION_NAME = args.collection
    if args.file_type_limits is not None:
        Config.FILE_TYPE_LIMITS = {}
        for ft_limit in args.file_type_limits:
            try:
                ft, limit = ft_limit.split('=')
                Config.FILE_TYPE_LIMITS[ft] = int(limit)
            except ValueError:
                logging.warning(f"Ignoring invalid file_type_limit format: {ft_limit}")

    # Handle Proxy Configuration
    if args.proxy:
        # For internetarchive to use the proxy, set it in the session's proxies
        # This has already been handled in the download_collection_chunk function
        pass  # No action needed here since it's handled in the function

    # Check if downloads_dir exists
    if not os.path.isdir(Config.BASE_DIR):
        print(f"Error: The directory '{Config.BASE_DIR}' does not exist.")
        logging.error(f"The directory '{Config.BASE_DIR}' does not exist.")
        return

    # Start processing folders
    download_collection_chunk(start, end, proxy=args.proxy)

if __name__ == "__main__":
    main()