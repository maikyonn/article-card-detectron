"""
This is a helper file used by 0_download-launcher.py to download items from Internet Archive collections.
Please run 0_download-launcher.py instead of running this file directly.
"""

import os
import argparse
import internetarchive as ia
from tqdm import tqdm
import logging
import time
from requests.exceptions import Timeout, ConnectionError, RequestException

logging.basicConfig(
    filename='download_log.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_directory_if_needed(identifier, base_dir):
    item_dir = os.path.join(base_dir, identifier)
    os.makedirs(item_dir, exist_ok=True)
    return item_dir

def retry_operation(func, max_retries, retry_delay, *args, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except (Timeout, ConnectionError, RequestException) as e:
            retries += 1
            logging.warning(f"Attempt {retries}/{max_retries} failed with error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Unexpected error: {e}. No retry.")
            break
    logging.error(f"Operation failed after {max_retries} retries.")
    return None

def download_collection_chunk(start, end, proxy=None, max_retries=5, retry_delay=5, base_dir='downloads-16k-proxy', 
                            collection_name='news-homepages', file_type_limits=None, download_timeout=30):
    if file_type_limits is None:
        file_type_limits = {'.html': 15, 'fullpage.jpg': 15}

    try:
        collection_subset = []

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

        search_query = f'collection:{collection_name}'
        search = ia.search_items(search_query)

        logging.info("Fetching search results from Internet Archive...")
        with tqdm(desc="Fetching collection items", unit="item") as pbar:
            for result in search:
                collection_subset.append(result)
                pbar.update(1)
                if end is not None and len(collection_subset) >= end:
                    break

        if end is not None:
            collection_subset = collection_subset[start:end]
        else:
            collection_subset = collection_subset[start:]

        total_items = len(collection_subset)

        if total_items == 0:
            logging.info("No items found in the specified range.")
            print("No items found in the specified range.")
            return

        with tqdm(total=total_items, desc="Processing items from collection") as pbar:
            for result in collection_subset:
                identifier = result.get('identifier', 'unknown_identifier')
                logging.info(f"Processing item: {identifier}")
                print(f"\nProcessing item: {identifier}")

                item = retry_operation(ia.get_item, max_retries, retry_delay, identifier)
                if item is None:
                    logging.error(f"Failed to fetch item details for {identifier}. Skipping.")
                    pbar.update(1)
                    continue

                try:
                    files = list(item.get_files())
                except Exception as e:
                    logging.error(f"Error fetching files for {identifier}: {e}. Skipping.")
                    print(f"Error fetching files for {identifier}: {e}. Skipping.")
                    pbar.update(1)
                    continue

                file_counts = {ft: 0 for ft in file_type_limits}
                filtered_files = []

                for file in files:
                    for file_type, limit in file_type_limits.items():
                        if file.name.endswith(file_type) and file_counts[file_type] < limit:
                            filtered_files.append(file)
                            file_counts[file_type] += 1
                    # Check if all limits are reached
                    if all(file_counts[ft] >= limit for ft, limit in file_type_limits.items()):
                        break

                # Only proceed if required file types are found
                if all(count > 0 for count in file_counts.values()):
                    logging.info(f"Downloading {len(filtered_files)} files for {identifier}.")
                    print(f"Downloading {len(filtered_files)} files for {identifier}.")

                    # Create directory only if there are files to download
                    item_dir = create_directory_if_needed(identifier, base_dir)

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
                            for attempt in range(1, max_retries + 1):
                                try:
                                    # Set a timeout for the download operation if supported
                                    file.download(destdir=item_dir, timeout=download_timeout)
                                    success = True
                                    break
                                except (Timeout, ConnectionError, RequestException) as e:
                                    logging.warning(f"Attempt {attempt}/{max_retries} failed to download {file.name}: {e}. Retrying in {retry_delay} seconds...")
                                    time.sleep(retry_delay)
                                except Exception as e:
                                    logging.error(f"Unexpected error while downloading {file.name}: {e}. No retry.")
                                    break
                            if not success:
                                logging.error(f"Failed to download {file.name} after {max_retries} attempts. Skipping.")
                                print(f"Failed to download {file.name} after {max_retries} attempts. Skipping.")

                            # Update the file progress bar
                            file_pbar.update(1)

                    logging.info(f"Verifying downloaded files for {identifier}.")
                    print(f"Verifying downloaded files for {identifier}.")

                    try:
                        actual_files = os.listdir(item_dir)
                        verification_passed = True
                        for file_type, limit in file_type_limits.items():
                            actual_count = len([f for f in actual_files if f.endswith(file_type)])
                            expected_count = file_type_limits[file_type]
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


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Download a chunk of a collection from Internet Archive through a proxy.")
    parser.add_argument('--start', type=int, default=0, help='Start index of the collection chunk (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index of the collection chunk (exclusive)')
    parser.add_argument('--max_retries', type=int, default=5, help='Maximum number of retries for operations')
    parser.add_argument('--retry_delay', type=int, default=5, help='Delay between retries in seconds')
    parser.add_argument('--base_dir', type=str, default='downloads-16k-proxy', help='Base directory for downloads')
    parser.add_argument('--collection', type=str, default='news-homepages', help='Name of the collection to download from')
    parser.add_argument('--file_type_limits', type=str, nargs='*', help='File type limits in the format type=limit, e.g., .html=5 fullpage.jpg=2')
    parser.add_argument('--download_timeout', type=int, default=30, help='Download timeout in seconds')
    parser.add_argument('--proxy', type=str, help='Proxy server in the format socks5://username:password@IP:Port')

    args = parser.parse_args()

    # Process file type limits if provided
    file_type_limits = {'.html': 15, 'fullpage.jpg': 15}  # Default values
    if args.file_type_limits:
        file_type_limits = {}
        for ft_limit in args.file_type_limits:
            try:
                ft, limit = ft_limit.split('=')
                file_type_limits[ft] = int(limit)
            except ValueError:
                logging.warning(f"Ignoring invalid file_type_limit format: {ft_limit}")

    # Check if downloads_dir exists
    if not os.path.isdir(args.base_dir):
        print(f"Error: The directory '{args.base_dir}' does not exist.")
        logging.error(f"The directory '{args.base_dir}' does not exist.")
        return

    # Start processing folders
    download_collection_chunk(
        start=args.start,
        end=args.end,
        proxy=args.proxy,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        base_dir=args.base_dir,
        collection_name=args.collection,
        file_type_limits=file_type_limits,
        download_timeout=args.download_timeout
    )

if __name__ == "__main__":
    main()