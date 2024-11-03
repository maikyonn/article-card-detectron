import os

def count_csv_and_folders(directory):
    csv_count = 0
    folder_count = 0

    # Walk through all files and folders in the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        # Count folders
        folder_count += len(dirs)
        
        # Count .csv files
        csv_count += sum(1 for file in files if file.endswith('.csv'))

    return csv_count, folder_count

# Set the path to the directory you want to search
directory_path = 'v5-proxy'  # Replace with your directory path

# Get the counts
csv_count, folder_count = count_csv_and_folders(directory_path)

# Print the results
print(f"Number of .csv files: {csv_count}")
print(f"Number of folders: {folder_count}")