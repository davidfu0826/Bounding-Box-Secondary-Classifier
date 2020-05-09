import argparse
import os
import sys
import glob
from pathlib import Path
from typing import List

import PIL.Image as Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.models import get_model
from utils.data import get_test_transforms

def get_image_paths(img_dir: str) -> List[str]:
    """Reads all images in given directory. (Assuming .jpg)
    
    Args:
        img_dir: Path to directory
    """
    return glob.glob(os.path.join(img_dir) + "/*.jpg")

def get_inference_engine(weights_path, num_classes):
    model = get_model(num_classes)
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    class_to_idx = state_dict["class_to_idx"]
    idx_to_class = {i: label for i, label in enumerate(class_to_idx)}
    return model, idx_to_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using your trained model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument("--img-dir", type=str, default="data/unlabelled_images",
                        help="path to directory with unlabelled images")
    parser.add_argument("--img-size", type=int, default=32,
                        help="image size when using inference")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="number of classes")
    parser.add_argument("--weights", type=str, default="weights/best.pt",
                        help="path to weights file (.pt)")

    args = parser.parse_args()
    img_dir = args.img_dir
    img_size = args.img_size
    num_classes = args.num_classes
    weights_path = args.weights
    
    model, idx_to_class = get_inference_engine(weights_path, num_classes)
    transform = get_test_transforms(img_size)
    img_paths = get_image_paths(img_dir)
    
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/secondary_predictions.txt", "w") as f:
        for img_path in tqdm(img_paths):
            
            # Read image
            img = Image.open(img_path)
            
            # Pre-processing
            X = transform(img)
            X = X.unsqueeze(0)
            
            # Inference
            pred = model(X)
            
            # Post-processing
            pred_class_id = int(pred.argmax())
            pred_class = idx_to_class[pred_class_id]
            
            probability = F.softmax(pred)[0][pred_class_id]
            
            # Write to file
            f.write(f"{os.path.basename(img_path)} {pred_class} {probability}\n")