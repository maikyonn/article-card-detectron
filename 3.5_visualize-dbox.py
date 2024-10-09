import os
import csv
import cv2
import argparse

def visualize_and_save_bounding_boxes(input_directory, output_directory):
    """
    Scans the input directory for JPG images and their corresponding CSV files containing
    bounding box coordinates. It then visualizes these bounding boxes on the images and
    saves the resulting images to the output directory.

    :param input_directory: Path to the directory containing JPG and CSV files.
    :param output_directory: Path to the directory where processed images will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(input_directory, filename)
            
            # Remove the last two extensions: .fullpage and .jpg
            base = filename.rsplit('.', 2)[0]
            csv_filename = base + '.csv'
            csv_path = os.path.join(input_directory, csv_filename)

            # Check if the corresponding CSV file exists
            if os.path.exists(csv_path):
                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                # Read bounding box data from CSV
                with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    boxes_drawn = False  # Flag to check if any boxes are drawn
                    for row in reader:
                        try:
                            # Extract x, y, width, height from the row
                            x = float(row['x'])
                            y = float(row['y'])
                            width = float(row['width'])
                            height = float(row['height'])

                            # Skip rows with zero width or height
                            if width == 0 or height == 0:
                                continue

                            # Compute x_max and y_max
                            x_min = int(round(x))
                            y_min = int(round(y))
                            x_max = int(round(x + width))
                            y_max = int(round(y + height))

                            # Draw rectangle on the image
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            boxes_drawn = True
                        except (ValueError, KeyError) as e:
                            print(f"Invalid row in CSV {csv_path}: {row}. Error: {e}")
                            continue

                if boxes_drawn:
                    # Define the output path
                    output_path = os.path.join(output_directory, filename)
                    # Save the processed image
                    success = cv2.imwrite(output_path, image)
                    if success:
                        print(f"Saved processed image to: {output_path}")
                    else:
                        print(f"Failed to save image to: {output_path}")
                else:
                    print(f"No valid bounding boxes found in CSV for image: {filename}")
            else:
                print(f"No corresponding CSV file for image: {filename}")

    print("Processing completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize bounding boxes from CSV and save images.')
    parser.add_argument('--input', required=True, help='Path to the input directory containing JPG and CSV files.')
    parser.add_argument('--output', required=True, help='Path to the output directory to save processed images.')
    args = parser.parse_args()

    visualize_and_save_bounding_boxes(args.input, args.output)