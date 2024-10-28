import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# Define the number of worker threads
NUM_THREADS = 64

# Lock for thread-safe printing
print_lock = threading.Lock()

def duplicate_folder_multithreaded(source, destination, exclude_phrase, num_threads=32):
    """
    Duplicate a folder using multithreading with verbosity and a progress bar,
    excluding files that contain a specific phrase in their filenames.

    Parameters:
    - source (str): Path to the source folder.
    - destination (str): Path to the destination folder.
    - exclude_phrase (str): Phrase to exclude files containing it in their names.
    - num_threads (int): Number of worker threads to use.
    """

    # Step 1: Gather all files to copy
    files_to_copy = []
    for root, dirs, files in os.walk(source):
        # Compute the relative path from the source directory
        rel_path = os.path.relpath(root, source)
        for file in files:
            if exclude_phrase not in file:
                source_file = os.path.join(root, file)
                dest_dir = os.path.join(destination, rel_path)
                dest_file = os.path.join(dest_dir, file)
                files_to_copy.append((source_file, dest_file))

    total_files = len(files_to_copy)
    if total_files == 0:
        print(f"No files to copy. All files contain the phrase '{exclude_phrase}'.")
        return

    print(f"Starting duplication of '{source}' to '{destination}'.")
    print(f"Total files to copy: {total_files}\n")

    # Step 2: Create all necessary directories in the destination
    # This prevents multiple threads from attempting to create directories simultaneously
    directories = set()
    for _, dest_file in files_to_copy:
        dest_dir = os.path.dirname(dest_file)
        directories.add(dest_dir)

    for dest_dir in sorted(directories):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            with print_lock:
                print(f"Ensured directory exists: {dest_dir}")
        except Exception as e:
            with print_lock:
                print(f"Failed to create directory '{dest_dir}'. Error: {e}")

    # Step 3: Define the file copy function
    def copy_file(source_dest_tuple):
        source_file, dest_file = source_dest_tuple
        try:
            shutil.copy2(source_file, dest_file)
            with print_lock:
                print(f"Copied: '{source_file}' -> '{dest_file}'")
            return True
        except Exception as e:
            with print_lock:
                print(f"Failed to copy '{source_file}'. Error: {e}")
            return False

    # Step 4: Copy files using ThreadPoolExecutor with a progress bar
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Initialize the progress bar
        with tqdm(total=total_files, desc="Copying files", unit="file") as pbar:
            # Submit all copy tasks
            future_to_file = {executor.submit(copy_file, file_pair): file_pair for file_pair in files_to_copy}
            
            # Iterate over the completed futures
            for future in as_completed(future_to_file):
                result = future.result()
                pbar.update(1)

    print(f"\nDuplication completed successfully to '{destination}' excluding files with '{exclude_phrase}'.")

if __name__ == "__main__":
    # Define source and destination folder names
    source_folder = "data/v3-combined"
    destination_folder = "data/v4-combined-only-new"  # You can change this as needed
    phrase_to_exclude = "-old-best"

    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist. Please check the path and try again.")
    else:
        duplicate_folder_multithreaded(
            source=source_folder,
            destination=destination_folder,
            exclude_phrase=phrase_to_exclude,
            num_threads=NUM_THREADS
        )