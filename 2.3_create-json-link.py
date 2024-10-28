import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

def setup_logging():
    logging.basicConfig(
        filename='link_extraction.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def extract_links(args):
    html_file, input_dir, output_dir = args
    try:
        # Determine the relative path of the HTML file with respect to the input directory
        relative_path = os.path.relpath(html_file, input_dir)
        # Change the extension to 'hyperlinks.json'
        json_relative_path = os.path.splitext(relative_path)[0] + '.hyperlinks.json'
        # Construct the full path for the JSON file in the output directory
        json_file = os.path.join(output_dir, json_relative_path)
        # Create the necessary subdirectories in the output directory
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        # Read and parse the HTML file
        with open(html_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        
        # Extract links and their corresponding text
        links = [
            {'href': a['href'], 'text': a.get_text(strip=True)}
            for a in soup.find_all('a', href=True)
            if len(a.get_text(strip=True).split()) >= 3
        ]
        
        # Write the extracted links to the JSON file
        with open(json_file, 'w', encoding='utf-8') as outfile:
            json.dump(links, outfile, indent=4)
        
        logging.info(f'Successfully processed {html_file} → {json_file}')
    except Exception as e:
        logging.error(f'Error processing {html_file}: {e}')

def main(input_dir, output_dir):
    setup_logging()
    # Gather all .html files from the input directory and its subdirectories
    html_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files if file.lower().endswith('.html')
    ]
    # Prepare arguments for each thread
    args_list = [(html_file, input_dir, output_dir) for html_file in html_files]
    
    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(extract_links, args_list)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract links from HTML files into JSON.')
    parser.add_argument('input_directory', help='Directory to process')
    parser.add_argument('output_directory', help='Directory to write JSON files')
    args = parser.parse_args()
    main(args.input_directory, args.output_directory)

