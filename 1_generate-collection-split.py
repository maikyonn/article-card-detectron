import random

# Load the list from the provided text file
with open('filtered_identifiers.txt', 'r') as file:
    lines = file.read().splitlines()

# Shuffle the list randomly
random.shuffle(lines)

# Define the number of files to split into
num_files = 16

# Calculate the number of items per file
chunk_size = len(lines) // num_files
remainder = len(lines) % num_files

# Split the list into the specified number of files
start = 0
for i in range(num_files):
    # Calculate end index for the current chunk
    end = start + chunk_size + (1 if i < remainder else 0)
    # Get the current chunk of data
    chunk = lines[start:end]
    # Write the chunk to a new file
    with open(f'news_list_part_{i+1}.txt', 'w') as output_file:
        output_file.write('\n'.join(chunk))
    # Update start index for the next chunk
    start = end

print(f"Successfully split into {num_files} files.")