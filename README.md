# Data Processing and Visualization Repository
## Overview

This repository presents a comprehensive workflow for training a Detectron2 model tailored to identify and classify article cards on news homepages. The project initiates with the acquisition of web data through proxy-configured download scripts, ensuring efficient and scalable data collection. Following data acquisition, scripts are employed to extract and generate precise bounding boxes from HTML files, facilitating accurate localization of article regions within the webpages. The dataset is then methodically split and pruned using specialized scripts to enhance quality and relevance, preparing it for effective model training. Conversion tools transform the curated data into the COCO format, a standard compatible with Detectron2, thereby streamlining the training process. Advanced visualization scripts are included to compare ground truth annotations with model predictions, providing valuable insights into the model's performance and areas for improvement. Additionally, utility scripts support tasks such as data validation, file management, and bounding box manipulation, ensuring the integrity and organization of the dataset. By integrating these diverse scripts into a unified pipeline, this repository offers an end-to-end solution for developing a robust object detection model, ultimately aiming to enhance the automated analysis and organization of content on news websites.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [0. Download Launcher](#0-download-launcher)
  - [0.5. Archive Downloader Proxy](#05-archive-downloader-proxy)
  - [1. Generate Collection Split](#1-generate-collection-split)
  - [2. Generate Bounding Boxes](#2-generate-bounding-boxes)
  - [3. Generate Cards and Comparison](#3-generate-cards-and-comparison)
  - [3.5. Visualize Bounding Boxes](#35-visualize-bounding-boxes)
  - [4. Summarize Collections](#4-summarize-collections)
  - [4.5. Prune CSVs](#45-prune-csvs)
  - [5. Create COCO Dataset](#5-create-coco-dataset)
  - [6. Train Detectron2 Model](#6-train-detectron2-model)
  - [7. Visualize Predictions](#7-visualize-predictions)
  - [7.5. Targeted Visualization](#75-targeted-visualization)
  - [8. Verify Downloads](#8-verify-downloads)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains a comprehensive suite of Python scripts designed for efficient data processing, visualization, and management tasks. The scripts are organized in a specific execution order to streamline workflows involving downloading data, generating and pruning datasets, training machine learning models, and visualizing results.

## Features

- **Downloading Tools**: Efficiently download and manage datasets using proxy configurations.
- **Data Splitting and Processing**: Split large datasets into manageable chunks and generate bounding boxes for images.
- **Visualization Tools**: Visualize ground truth annotations and model predictions with customizable parameters.
- **Data Summarization and Pruning**: Summarize dataset statistics and prune OCR results based on similarity scores.
- **COCO Dataset Creation**: Convert datasets into COCO format for compatibility with Detectron2.
- **Model Training**: Train Detectron2 models for object detection and instance segmentation.
- **Verification Tools**: Validate and clean up downloaded data to ensure integrity.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

Ensure the following Python packages are installed:

- `json`
- `cv2` (OpenCV)
- `matplotlib`
- `pathlib`
- `os`
- `collections`
- `random`
- `torch`
- `detectron2`
- `argparse`
- `logging`
- `concurrent.futures`
- `tqdm`
- `Pillow`
- `BeautifulSoup4`
- `scipy`
- `numpy`
- `pandas`
- `more-itertools`
- `Playwright`

You can install all dependencies using the provided `requirements.txt` file:
bash
pip install -r requirements.txt


## Usage

The scripts are intended to be run in the following order to ensure a smooth and efficient workflow.

### 0. Download Launcher

**Script:** `0_download-launcher.py`

**Description:**  
Launches multiple instances of the download script with different proxies to efficiently download datasets.

**Usage:**
bash
python 0_download-launcher.py


**Functionality:**
- Reads proxies from `socks5.txt`.
- Splits the download task into chunks based on the number of proxies.
- Launches download scripts concurrently with a slight delay to prevent system overload.

---

### 0.5. Archive Downloader Proxy

**Script:** `0.5_archive-downloader-proxy.py`

**Description:**  
Handles the downloading of archives using proxy configurations to ensure smooth and uninterrupted downloads.

**Usage:**
bash
python 0.5_archive-downloader-proxy.py


**Functionality:**
- Manages proxy settings.
- Downloads archives required for dataset preparation.
- Ensures downloaded files are archived appropriately.

---

### 1. Generate Collection Split

**Script:** `1_generate-collection-split.py`

**Description:**  
Shuffles a list of identifiers and splits them into multiple text files for batch processing.

**Usage:**
bash
python 1_generate-collection-split.py


**Functionality:**
- Reads `filtered_identifiers.txt`.
- Shuffles the entries randomly.
- Splits the list into 16 parts named `news_list_part_1.txt` to `news_list_part_16.txt`.

---

### 2. Generate Bounding Boxes

**Script:** `2_generate-dboxes.py`

**Description:**  
Processes HTML files to extract bounding boxes using Playwright for browser automation and asyncio for concurrency. Saves the results as CSV files.

**Usage:**
bash
python 2_generate-dboxes.py \
--download-dir /path/to/downloads_directory \
--max-concurrent-tasks 60


**Arguments:**
- `--download-dir`: Path to the directory containing HTML files to be processed.
- `--max-concurrent-tasks`: Maximum number of HTML files to process concurrently. *(Default: 60)*

**Functionality:**
- Initializes Playwright and launches a headless browser.
- Extracts bounding boxes from HTML files.
- Saves bounding box data to CSV files.
- Manages concurrency to optimize processing speed.

---

### 3. Generate Cards and Comparison

**Script:** `3_generate-cards-and-comparison.py`

**Description:**  
Crops images based on bounding boxes, performs OCR, validates against JSON data, and saves the results.

**Usage:**
bash
python 3_generate-cards-and-comparison.py \
--main_dir /path/to/main_directory \
--output_dir /path/to/output_directory \
--languages en fr \
--use_gpu


**Arguments:**
- `--main_dir`: Path to the main directory containing subdirectories with triplet files.
- `--output_dir`: Path to the directory where cropped images and OCR results will be saved.
- `--use_gpu`: Flag to enable GPU usage for OCR. *(Default: CPU)*

**Functionality:**
- Crops images based on bounding box data.
- Performs Optical Character Recognition (OCR) on cropped images.
- Validates OCR results against JSON data.
- Saves pruned and validated OCR results.

---

### 3.5. Visualize Bounding Boxes

**Script:** `3.5_visualize-dbox.py`

**Description:**  
Visualizes bounding boxes on images by drawing rectangles based on CSV annotation files.

**Usage:**
python 3.5_visualize-dbox.py \
--input /path/to/input_directory \
--output /path/to/output_directory \
--use-gpu


**Arguments:**
- `--input`: Path to the input directory containing JPG and CSV files.
- `--output`: Path to the output directory to save processed images with bounding boxes.

**Functionality:**
- Scans the input directory for JPG images and corresponding CSV files.
- Draws bounding boxes on images based on CSV annotations.
- Saves the annotated images to the output directory.

---

### 4. Summarize Collections

**Script:** `4_summarize-collections.py`

**Description:**  
Summarizes similarity statistics of all collections by calculating average similarity scores and distribution percentages. Ranks collections and exports the top N collections based on performance.

**Usage:**
bash
python 4_summarize-collections.py \
--output_dir /path/to/output_directory \
--summary_file collections_summary.txt \
--top_n 500 \
--top_collections_file top_500_collections.txt


**Arguments:**
- `--output_dir`: Path to the main output directory containing subdirectories for each collection.
- `--summary_file`: Filename for the summary text file. *(Default: collections_summary.txt)*
- `--top_n`: Number of top collections to export based on average similarity score. *(Default: 500)*
- `--top_collections_file`: Filename for the top N collections text file. *(Default: top_500_collections.txt)*

**Functionality:**
- Iterates through collection directories.
- Calculates weighted average similarity scores.
- Determines the distribution of similarity categories (high, medium, low, very low).
- Generates a comprehensive summary and exports the top collections.

---

### 4.5. Prune CSVs

**Script:** `4.5_prune-csvs.py`

**Description:**  
Prunes OCR result CSV files based on similarity scores. Adds necessary columns and retains entries based on threshold or percentile methods.

**Usage:**
bash
Using threshold method
python 4.5_prune-csvs.py \
--input_dir /path/to/original_csvs \
--output_dir /path/to/pruned_csvs \
--prune_method threshold \
--similarity_threshold 0.8


**Arguments:**
- `--input_dir`: Directory containing the original OCR CSV files.
- `--output_dir`: Directory where pruned CSV files will be saved.
- `--prune_method`: Method to define "good" scores (`threshold` or `percentile`).
- `--similarity_threshold`: *(If `prune_method` is `threshold`)* The minimum similarity score to retain. *(Default: 0.8)*
- `--percentile`: *(If `prune_method` is `percentile`)* The percentile threshold to retain top entries. *(Default: 80.0)*

**Functionality:**
- Adds `matched_text` and `weighted_similarity_score` columns if missing.
- Prunes entries based on the specified method.
- Saves the pruned CSV files to the output directory.

---

### 5. Create COCO Dataset

**Script:** `5_create-coco.py`

**Description:**  
Converts a directory of collections with images and CSV annotations into the COCO format for Detectron2. Supports dataset splitting into training and validation sets.

**Usage:**
python 5_create-coco.py \
--downloads_dir /path/to/downloads_directory \
--dataset_dir /path/to/dataset_directory \
--top_collections_file top_500_collections.txt \
--images_dir_name images \
--annotations_dir_name annotations \
--train_annotations_file train_annotations.json \
--val_annotations_file val_annotations.json \
--categories object \
--split_ratio 0.8 \
--random_seed 42\


**Arguments:**
- `--downloads_dir`: Path to the downloads directory.
- `--dataset_dir`: Path to the dataset directory where COCO-formatted data will be saved.
- `--top_collections_file`: Path to the text file containing names of top collections to include.
- `--images_dir_name`: Directory name for images within each collection. *(Default: images)*
- `--annotations_dir_name`: Directory name for annotations within each collection. *(Default: annotations)*
- `--train_annotations_file`: Filename for the training annotations JSON. *(Default: train_annotations.json)*
- `--val_annotations_file`: Filename for the validation annotations JSON. *(Default: val_annotations.json)*
- `--categories`: Categories present in the dataset. *(Example: object)*
- `--split_ratio`: Ratio to split data into training and validation sets. *(Default: 0.8)*
- `--random_seed`: Random seed for reproducibility. *(Default: 42)*

**Functionality:**
- Iterates through top collections specified in the text file.
- Converts image and bounding box data into COCO format.
- Splits the dataset into training and validation sets based on the split ratio.
- Saves the COCO-formatted JSON files for training and validation.

---

### 6. Train Detectron2 Model

**Script:** `6_train-detectron.py`

**Description:**  
Trains a Detectron2 model using the COCO-formatted dataset. Supports distributed training and integrates with Weights & Biases for experiment tracking.

**Usage:**

bash
python 6_train-detectron.py \
--data-dir /path/to/data_directory \
--output-dir /path/to/output_directory \
--wandb-run-name "detectron2_training_run"


**Arguments:**
- `--data-dir`: Path to the data directory containing COCO-formatted datasets.
- `--output-dir`: Path to the output directory where trained models and logs will be saved.
- `--wandb-run-name`: Name for the Weights & Biases run for experiment tracking.

**Functionality:**
- Sets up the Detectron2 configuration.
- Initializes training with specified parameters.
- Implements early stopping based on validation performance.
- Logs training metrics to Weights & Biases.
- Saves the final trained model weights.

---

### 7. Visualize Predictions

**Script:** `7_visualize-predictions.py`

**Description:**  
Visualizes model predictions on all images in a specified directory using Detectron2. Draws predicted bounding boxes and saves the annotated images.

**Usage:**
bash
python 7_visualize-predictions.py \
--images_dir /path/to/images \
--output_pred_dir /path/to/output_predictions \
--model_weights /path/to/model_weights.pth \
--score_threshold 0.5


**Arguments:**
- `--images_dir`: Path to the directory containing images to process.
- `--output_pred_dir`: Directory to save images with predicted bounding boxes.
- `--model_weights`: Path to the trained Detectron2 model weights.
- `--score_threshold`: Minimum score for the predicted bounding boxes to be visualized. *(Default: 0.5)*

**Functionality:**
- Loads the trained Detectron2 model.
- Processes each image in the specified directory.
- Draws bounding boxes on predictions above the score threshold.
- Saves the annotated images to the output directory.

---

### 7.5. Targeted Visualization

**Script:** `7_targeted-visualize.py`

**Description:**  
Advanced visualization tool for ground truth annotations and model predictions with enhanced customization options. Supports selecting specific samples and adjusting visualization parameters.

**Usage:**
bash
python 7_targeted-visualize.py \
--annotations_json /path/to/val_annotations.json \
--images_dir /path/to/images \
--output_gt_dir /path/to/output_gt \
--output_pred_dir /path/to/output_pred \
--num_samples 30 \
--model_weights /path/to/model_weights.pth \
--score_threshold 0.5


**Arguments:**
- `--annotations_json`: Path to the COCO annotations JSON file.
- `--images_dir`: Path to the directory containing images.
- `--output_gt_dir`: Directory to save images with ground truth bounding boxes.
- `--output_pred_dir`: Directory to save images with predicted bounding boxes.
- `--num_samples`: Number of samples to visualize. *(Default: 30)*
- `--model_weights`: Path to the trained Detectron2 model weights. If not provided, predictions are skipped.
- `--score_threshold`: Minimum score for the predicted bounding boxes to be visualized. *(Default: 0.5)*

**Functionality:**
- Selects random samples from the dataset.
- Visualizes both ground truth and predicted bounding boxes.
- Saves the annotated images to specified directories for comparison.

---


- **Scripts:**
  - `0_download-launcher.py`: Launches download scripts with proxy configurations.
  - `0.5_archive-downloader-proxy.py`: Downloads archives using proxies.
  - `1_generate-collection-split.py`: Splits identifier lists for batch processing.
  - `2_generate-dboxes.py`: Extracts bounding boxes from HTML files.
  - `3_generate-cards-and-comparison.py`: Crops images, performs OCR, and compares results.
  - `3.5_visualize-dbox.py`: Visualizes bounding boxes on images.
  - `4_summarize-collections.py`: Summarizes similarity statistics of collections.
  - `4.5_prune-csvs.py`: Prunes OCR CSV files based on similarity scores.
  - `5_create-coco.py`: Converts datasets to COCO format.
  - `6_train-detectron.py`: Trains Detectron2 models.
  - `7_visualize-predictions.py`: Visualizes model predictions on images.
  - `7_targeted-visualize.py`: Advanced visualization of ground truth and predictions.
  - `8_verify-download.py`: Validates and cleans up downloaded data.
  - `box_utils.py`: Utilities for bounding box operations.
  - `del-csv-jpg.py`: Deletes specified CSV and JPG files.
  
- **JavaScript Files (`js/`):**
  - `model_utils.js`: Utility scripts for models.
  - `utils.js`: General utility scripts.
  - `psl.min.js`: URL parsing library.
  - `trained_lr_obj.json`: Trained model weights.

- **Configuration:**
  - `requirements.txt`: Lists all Python dependencies.
  - `README.md`: This documentation file.
  - `LICENSE`: Project license file.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**
   ```bash
   git commit -m "Add your message here"
   ```

4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

Please ensure your code adheres to the project's coding standards and includes necessary documentation.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com).