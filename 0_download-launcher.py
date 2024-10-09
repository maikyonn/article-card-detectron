import subprocess
import time

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

def launch_scripts(proxies, script_path, start_indices, end_indices):
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
            '--proxy', proxy
        ]
        print(f"Launching script with proxy {proxy}, start={start}, end={end}")
        process = subprocess.Popen(cmd)
        processes.append(process)
        time.sleep(10)  # Slight delay to prevent overwhelming the system
    return processes

def main():
    proxies_file = 'socks5.txt'
    download_script = '0.5_archive-downloader-proxy.py'  # Replace with your script's filename

    # Read proxies from file
    proxies = read_proxies(proxies_file)
    if not proxies:
        print("No proxies found in the proxies file.")
        return

    # Define how to split your collection into chunks
    # For example, each proxy handles a chunk of 100 items
    chunk_size = 200
    start_indices = [i * chunk_size for i in range(len(proxies))]
    end_indices = [(i + 1) * chunk_size for i in range(len(proxies))]

    # Launch scripts with different proxies
    processes = launch_scripts(proxies, download_script, start_indices, end_indices)

    # Optionally, wait for all processes to complete
    for process in processes:
        process.wait()

    print("All download processes have been completed.")

if __name__ == "__main__":
    main()