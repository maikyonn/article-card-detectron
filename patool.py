import os
import tarfile
import sys
import threading
import queue
import io
import argparse
from concurrent.futures import ThreadPoolExecutor

def get_all_files(input_folder):
    """
    Recursively retrieves all file paths within the input folder.
    """
    file_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, start=input_folder)
            file_paths.append((full_path, relative_path))
    return file_paths

def compress_worker(file_queue, data_queue):
    """
    Worker thread to read file data and put it into the data queue.
    """
    while True:
        try:
            full_path, relative_path = file_queue.get_nowait()
        except queue.Empty:
            break
        try:
            with open(full_path, 'rb') as f:
                data = f.read()
            data_queue.put((relative_path, data))
        except Exception as e:
            print(f"Error reading {full_path}: {e}")
        finally:
            file_queue.task_done()

def compress_writer(tar_path, data_queue, total_files):
    """
    Writer thread to add files to the tar archive.
    """
    try:
        with tarfile.open(tar_path, "w") as tar:
            processed = 0
            while processed < total_files:
                try:
                    relative_path, data = data_queue.get(timeout=1)
                except queue.Empty:
                    continue  # Wait for more data

                try:
                    tarinfo = tarfile.TarInfo(name=relative_path)
                    tarinfo.size = len(data)
                    tar.addfile(tarinfo, fileobj=io.BytesIO(data))
                    processed += 1
                    if processed % 1000 == 0:
                        print(f"Compressed {processed}/{total_files} files...")
                except Exception as e:
                    print(f"Error writing {relative_path} to archive: {e}")
                finally:
                    data_queue.task_done()
    except Exception as e:
        print(f"Error opening tar file {tar_path}: {e}")

def compress_folder(input_folder, output_tar, num_threads=64):
    """
    Compresses the input_folder into output_tar using multiple threads.
    """
    # Validate input folder
    if not os.path.isdir(input_folder):
        print(f"Error: The input path '{input_folder}' is not a directory or does not exist.")
        sys.exit(1)

    # Get all files to be archived
    print("Scanning files...")
    all_files = get_all_files(input_folder)
    total_files = len(all_files)
    print(f"Total files to archive: {total_files}")

    if total_files == 0:
        print("No files to compress.")
        return

    # Initialize queues
    file_queue = queue.Queue()
    data_queue = queue.Queue(maxsize=1000)  # Adjust maxsize based on memory

    for file in all_files:
        file_queue.put(file)

    # Start writer thread
    writer_thread = threading.Thread(target=compress_writer, args=(output_tar, data_queue, total_files), daemon=True)
    writer_thread.start()

    # Start worker threads
    print("Starting worker threads for compression...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(num_threads):
            executor.submit(compress_worker, file_queue, data_queue)

    # Wait for all files to be processed
    file_queue.join()
    data_queue.join()

    print("Compression completed successfully.")

def decompress_worker(tar, extract_queue, output_directory):
    """
    Worker thread to extract file data and write to disk.
    """
    while True:
        try:
            tarinfo = extract_queue.get_nowait()
        except queue.Empty:
            break
        try:
            # Extract the file data
            file_obj = tar.extractfile(tarinfo)
            if file_obj is None:
                print(f"Skipping {tarinfo.name}, not a regular file.")
                extract_queue.task_done()
                continue
            data = file_obj.read()
            # Write the file data to disk
            output_path = os.path.join(output_directory, tarinfo.name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"Error extracting {tarinfo.name}: {e}")
        finally:
            extract_queue.task_done()

def decompress_tar(input_tar, output_directory, num_threads=64):
    """
    Decompresses the input_tar archive into output_directory using multiple threads.
    """
    # Validate input tar file
    if not os.path.isfile(input_tar):
        print(f"Error: The input tar file '{input_tar}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    try:
        with tarfile.open(input_tar, "r") as tar:
            # Get all file members
            all_members = [m for m in tar.getmembers() if m.isfile()]
            total_files = len(all_members)
            print(f"Total files to extract: {total_files}")

            if total_files == 0:
                print("No files to extract.")
                return

            # Initialize queue
            extract_queue = queue.Queue()
            for member in all_members:
                extract_queue.put(member)

            # Start worker threads
            print("Starting worker threads for decompression...")
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for _ in range(num_threads):
                    executor.submit(decompress_worker, tar, extract_queue, output_directory)

            # Wait for all files to be extracted
            extract_queue.join()

            print("Decompression completed successfully.")
    except Exception as e:
        print(f"Error opening tar file {input_tar}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Tar Compression and Decompression Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands: compress or decompress", required=True)

    # Compress sub-command
    compress_parser = subparsers.add_parser("compress", help="Compress a folder into a tar archive")
    compress_parser.add_argument("input_folder", type=str, help="Path to the input folder to compress")
    compress_parser.add_argument("output_tar", type=str, help="Path to the output tar archive")

    # Decompress sub-command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a tar archive into a folder")
    decompress_parser.add_argument("input_tar", type=str, help="Path to the input tar archive to decompress")
    decompress_parser.add_argument("output_directory", type=str, help="Path to the output directory")

    args = parser.parse_args()

    if args.command == "compress":
        compress_folder(args.input_folder, args.output_tar, num_threads=64)
    elif args.command == "decompress":
        input_tar = args.input_tar
        output_directory = args.output_directory
        decompress_tar(input_tar, output_directory, num_threads=64)
    else:
        parser.print_help()