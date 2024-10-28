#!/bin/bash

# Path to the text file containing folder names
file_path="/project/jonmay_1426/spangher/homepage-parser-latest/collection-stats/v1-stats/extracted_folder_names.txt"  # Replace with the actual path to your file

# Function to run the Python script for each folder
run_python_script() {
    folder_name="$1"
    python 7_visualize-predictions.py --file_filter "$folder_name" .
}

# Read the file line by line and run the Python script for each folder in the background
while IFS= read -r folder_name; do
    # Skip empty lines
    if [[ ! -z "$folder_name" ]]; then
        run_python_script "$folder_name" &
    fi
done < "$file_path"

# Wait for all background processes to finish
wait

echo "All tasks are completed."