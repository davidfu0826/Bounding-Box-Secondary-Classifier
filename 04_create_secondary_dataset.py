import os
import json
import argparse
from pathlib import Path
from shutil import copyfile, rmtree

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Post-process prediction results and create dataset using unlabelled data",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results", type=str, default="results/secondary_predictions.txt",
                        help="path to prediction results from secondary classifier.")
    parser.add_argument("--img-dir", type=str, default="data/unlabelled_images",
                        help="path to directory with unlabelled images")
    parser.add_argument("--create-dataset", action="store_true",
                        help="prepare unlabelled images as a new dataset using secondary classifier predictions")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="use only prediction with over <threshold> in probability")
    args = parser.parse_args()
    
    PROB_THRESHOLD = args.threshold
    
    with open(args.results) as f:
        lines = f.readlines()
        
        # 0. framename 1. secondary prediction 2. probability
        predictions = [line.replace("\n", "").split(" ") for line in lines]
       
    labels = set([pred for _, pred, _ in predictions])
    
    # Remove directory before creating the secondary dataset
    if os.path.isdir("data/secondary_dataset"):
        rmtree("data/secondary_dataset", ignore_errors=True)
    
    for label in labels:
        Path(f"data/secondary_dataset/{label}").mkdir(parents=True, exist_ok=True)
    
    for filename, pred, probability in tqdm(predictions):
        if float(probability) >= PROB_THRESHOLD:
            img_path = os.path.join(args.img_dir, filename)
            save_path = os.path.join("data/secondary_dataset", pred, filename)
            copyfile(img_path, save_path)
        
    for directory in os.listdir("data/secondary_dataset"):
        path = os.path.join("data/secondary_dataset", directory)
        print(f"{directory} - {len(os.listdir(path))}")
