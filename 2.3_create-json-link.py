"""
Link Extraction Tool

This script extracts links and their corresponding text from HTML files and saves them as JSON.
It processes HTML files recursively from an input directory and maintains the same directory
structure in the output directory. Only links with 3 or more words of text are included.

The script uses multithreading to process files concurrently for improved performance.

Features:
- Recursive HTML file processing
- Maintains source directory structure
- Multithreaded processing (32 workers)
- Filters links by minimum text length
- Comprehensive logging
- UTF-8 encoding support

Usage:
    python 2.3_create-json-link.py INPUT_DIR OUTPUT_DIR

Arguments:
    INPUT_DIR   Directory containing HTML files to process
    OUTPUT_DIR  Directory where JSON files will be saved

Output Format:
    Each HTML file will have a corresponding .hyperlinks.json file containing an array of:
    {
        "href": "link URL",
        "text": "link text content"
    }

Example:
    python 2.3_create-json-link.py ./downloads ./processed

    Input:  ./downloads/site1/page.html
    Output: ./processed/site1/page.hyperlinks.json
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import spacy

def setup_logging():
    logging.basicConfig(
        filename='link_extraction.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def is_valid_sentence(text, nlp):
    # Skip very short texts
    if len(text.split()) < 3:
        return False
        
    doc = nlp(text)
    
    # Check if the text contains at least one verb or noun
    has_verb_or_noun = any(token.pos_ in ['VERB', 'NOUN'] for token in doc)
    
    # Check if it's not just a date or number pattern
    not_just_date = not all(token.like_num or token.is_punct or token.is_space for token in doc)
    
    return has_verb_or_noun and not_just_date

def extract_links(args):
    html_file, input_dir, output_dir, nlp = args
    try:
        # Determine the relative path of the HTML file with respect to the input directory
        relative_path = os.path.relpath(html_file, input_dir)
        # Change the extension to 'hyperlinks.json'
        json_relative_path = os.path.splitext(relative_path)[0] + '-new.hyperlinks.json'
        # Construct the full path for the JSON file in the output directory
        json_file = os.path.join(output_dir, json_relative_path)
        # Create the necessary subdirectories in the output directory
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        # Read and parse the HTML file
        with open(html_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        
        # Extract links and their corresponding text with sentence validation
        links = [
            {'href': a['href'], 'text': a.get_text(strip=True)}
            for a in soup.find_all('a', href=True)
            if len(a.get_text(strip=True).split()) >= 3 
            and is_valid_sentence(a.get_text(strip=True), nlp)
        ]
        
        # Write the extracted links to the JSON file
        with open(json_file, 'w', encoding='utf-8') as outfile:
            json.dump(links, outfile, indent=4)
        
        logging.info(f'Successfully processed {html_file} → {json_file}')
    except Exception as e:
        logging.error(f'Error processing {html_file}: {e}')

def main(input_dir, output_dir):
    setup_logging()
    
    # Load spaCy model (you'll need to install it first)
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        logging.error("Please install spaCy model: python -m spacy download en_core_web_sm")
        return
    
    # Gather all .html files from the input directory and its subdirectories
    html_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files if file.lower().endswith('.html')
    ]
    # Prepare arguments for each thread
    args_list = [(html_file, input_dir, output_dir, nlp) for html_file in html_files]
    
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
