import cv2
from pathlib import Path
import os

import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Remove the register_datasets function

def visualize_predictions(
    images_dir,
    output_pred_dir,
    model_weights,
    score_threshold=0.5
):
    """
    Visualize model predictions on all images in the specified directory.
    
    Args:
        images_dir (str): Path to the directory containing images to process.
        output_pred_dir (str): Directory to save images with predicted bounding boxes.
        model_weights (str): Path to the trained Detectron2 model weights.
        score_threshold (float): Minimum score for the predicted bounding boxes to be visualized.
    """
    # Ensure output directory exists
    Path(output_pred_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove dataset registration
    
    # Initialize Detectron2 predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Change to your number of classes
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    predictor = DefaultPredictor(cfg)
    
    # Create a simple metadata catalog for visualization
    MetadataCatalog.get("custom_dataset").set(thing_classes=["object"])  # Replace "object" with your class name
    
    # Process all images in the directory
    for img_file in os.listdir(images_dir):
        img_path = Path(images_dir) / img_file
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue  # Skip non-image files
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image {img_path}")
            continue
        
        # Generate predictions
        outputs = predictor(image)
        
        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("custom_dataset"), scale=1.2, instance_mode=ColorMode.IMAGE)
        instances = outputs["instances"].to("cpu")
        instances = instances[instances.scores >= score_threshold]
        out = v.draw_instance_predictions(instances)
        
        pred_output_path = Path(output_pred_dir) / f"pred_{img_file}"
        cv2.imwrite(str(pred_output_path), out.get_image()[:, :, ::-1])
        print(f"Saved prediction image to {pred_output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Model Predictions.")
    parser.add_argument(
        "--images_dir", 
        type=str,  
        default="data/v3-combined/nytimes-2023",
        help="Path to the directory containing images to process."
    )
    parser.add_argument(
        "--output_pred_dir", 
        type=str, 
        default="generated-outputs/", 
        help="Directory to save images with predicted bounding boxes."
    )
    parser.add_argument(
        "--model_weights", 
        type=str, 
        default="data/v4-coco-finetune/model_final.pth", 
        help="Path to the trained Detectron2 model weights."
    )
    parser.add_argument(
        "--score_threshold", 
        type=float, 
        default=0.50, 
        help="Minimum score for the predicted bounding boxes to be visualized."
    )

    args = parser.parse_args()

    visualize_predictions(
        images_dir=args.images_dir,
        output_pred_dir=args.output_pred_dir,
        model_weights=args.model_weights,
        score_threshold=args.score_threshold
    )