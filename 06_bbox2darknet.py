import os
import argparse
import json
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="data/example/example_images",
                        help="Directory with the images where the bounding boxes are cropped from.")
    parser.add_argument("--output-dir", type=str, default="data/output/labels/",
                        help="Directory with the newly generated Darknet labels")
    parser.add_argument("--new-annot", type=str, default="data/results/new_annotations.json",
                        help="The newly generated annotation file (.json)")
    parser.add_argument("--old-annot", type=str, default="data/example/example_annotations.json",
                        help="The previously generated annotation file (.json)")
    parser.add_argument("--names", type=str, required=True, #default="../../YOLOv3 Projects/yolov3/jason_labels_sts.names",
                        help="Path to .names file containing labels")
    args = parser.parse_args()
    
    with open(args.old_annot) as f:
        inaccurate_labels = json.load(f)
    with open(args.new_annot) as f:
        accurate_labels = json.load(f)
        

    accuracies = list()
    num_labels = 0
    num_labels_changed = 0
    for img in inaccurate_labels:
        old_bboxes = inaccurate_labels[img]
        new_bboxes = accurate_labels[img]

        for i in range(len(old_bboxes)):
            old_bbox = old_bboxes[i]
            new_bbox = new_bboxes[i]
    
            num_labels += 1
            if old_bbox['label'] != new_bbox['label']:
                num_labels_changed += 1
                accuracies.append(new_bbox['accuracy'])

    print(f"{num_labels_changed} bounding box labels were changed.")
    print(f"The mean accuracy of the secondary accuracy was {np.mean(accuracies)}")
    print(f"The sandard deviation of the accuracy of the secondary accuracy was {np.std(accuracies)}")

    with open(args.names) as f:
        idx_to_class = [label.replace("\n", "") for label in f.readlines()[:-1]]
        
    class_to_idx = {label: idx for idx, label in enumerate(idx_to_class)}
    print(class_to_idx)
    
    counter = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for img_filename in tqdm(os.listdir(args.img_dir)):
        
        img = Image.open(os.path.join(args.img_dir, img_filename))
        width, height = img.size
        
        annot_file = img_filename.replace(".jpg", ".txt")
        with open(os.path.join(args.output_dir, annot_file), "w") as f:
            if accurate_labels.get(img_filename) is not None:
                for bbox in accurate_labels.get(img_filename):
                    tl = bbox["top-left"]
                    br = bbox["bottom-right"]
                    label = bbox["label"]
                    acc = bbox["accuracy"]

                    class_idx = class_to_idx[label]
                    norm_center_x = abs(tl[0] + br[0])/(2*width)
                    norm_center_y = abs(tl[1] + br[1])/(2*height)
                    norm_width = abs(tl[0] - br[0])/width
                    norm_height = abs(tl[1] - br[1])/height
                    
                    if acc > 0.9 or (label == "car" and acc > 0.3):
                        counter += 1
                        f.write(f"{class_idx} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n")
    print(f"We have {counter} bounding boxes.")
