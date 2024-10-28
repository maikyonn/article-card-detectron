import subprocess
import time
import argparse

def read_proxies(file_path):
    """
    Read proxies from the given file and return a list of proxy URLs.
    """
    proxies = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                ip, port, username, password = line.split(':')
                proxy_url = f'socks5://{username}:{password}@{ip}:{port}'
                proxies.append(proxy_url)
    return proxies

def launch_scripts(proxies, script_path, start_indices, end_indices, args):
    """
    Launch multiple instances of the download script with different proxies.
    """
    processes = []
    for proxy, start, end in zip(proxies, start_indices, end_indices):
        cmd = [
            'python',
            script_path,
            '--start', str(start),
            '--end', str(end),
            '--proxy', proxy,
            '--base_dir', args.base_dir,
            '--collection', args.collection,
            '--max_retries', str(args.max_retries),
            '--retry_delay', str(args.retry_delay),
            '--download_timeout', str(args.download_timeout)
        ]
        
        # Add file type limits if specified
        if args.file_type_limits:
            cmd.extend(['--file_type_limits'] + args.file_type_limits)
            
        print(f"Launching script with proxy {proxy}, start={start}, end={end}")
        process = subprocess.Popen(cmd)
        processes.append(process)
        time.sleep(args.delay)  # Configurable delay between launches
    return processes

def main():
    parser = argparse.ArgumentParser(description="Launch multiple instances of the download script with different proxies")
    parser.add_argument('--proxies_file', default='socks5.txt', help='Path to the proxies file')
    parser.add_argument('--script', default='0.5_archive-downloader-proxy.py', help='Path to the download script')
    parser.add_argument('--chunk_size', type=int, default=200, help='Number of items each proxy should handle')
    parser.add_argument('--delay', type=int, default=10, help='Delay between launching instances (seconds)')
    parser.add_argument('--base_dir', default='v5-proxy', help='Base directory for downloads')
    parser.add_argument('--collection', default='news-homepages', help='Collection name to download from')
    parser.add_argument('--max_retries', type=int, default=5, help='Maximum number of retries for operations')
    parser.add_argument('--retry_delay', type=int, default=5, help='Delay between retries in seconds')
    parser.add_argument('--download_timeout', type=int, default=30, help='Download timeout in seconds')
    parser.add_argument('--file_type_limits', nargs='*', help='File type limits (e.g., .html=15 fullpage.jpg=15)')
    
    args = parser.parse_args()

    # Read proxies from file
    proxies = read_proxies(args.proxies_file)
    if not proxies:
        print("No proxies found in the proxies file.")
        return

    # Calculate chunks based on the number of proxies
    start_indices = [i * args.chunk_size for i in range(len(proxies))]
    end_indices = [(i + 1) * args.chunk_size for i in range(len(proxies))]

    # Launch scripts with different proxies
    processes = launch_scripts(proxies, args.script, start_indices, end_indices, args)

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    print("All download processes have been completed.")

if __name__ == "__main__":
    main()
