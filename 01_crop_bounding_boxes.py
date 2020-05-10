import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

def get_image_paths(path_to_imgs: str) -> List[str]:
    """Return all paths to images given path to the directory
    
    Args: 
        path_to_imgs: Path to image directory
    """
    img_paths = glob.glob(path_to_imgs + os.sep + "*.jpg")
    return img_paths

def get_annotation_data(path_to_annot: str):
    """Return a dictionary corresponding to a json file given path to it.
    
    Args: 
        path_to_annot: Path to json file
    """
    with open(path_to_annot) as f:
        annotations = json.load(f)
        return annotations

def correct_annotation_format(annotations: Dict, nested_key:str = None):
    if nested_key is None:
        for img_filename in annotations:

            for bbox in annotations[img_filename]:
                pt1 = bbox.get("top-left")
                pt2 = bbox.get("bottom-right")
                
                if (pt1 is not None) and (pt2 is not None):
                    if isinstance(pt1[0], int) and isinstance(pt1[1], int) and isinstance(pt2[0], int) and isinstance(pt2[1], int):
                        return annotations
    else:
        annotations = annotations[nested_key]
        for img_filename in annotations:

            for bbox in annotations[img_filename]:
                pt1 = bbox.get("top-left")
                pt2 = bbox.get("bottom-right")
                
                if (pt1 is not None) and (pt2 is not None):
                    if isinstance(pt1[0], int) and isinstance(pt1[1], int) and isinstance(pt2[0], int) and isinstance(pt2[1], int):
                        return annotations
            
    
if __name__ == "__main__":

    description = """
    Crops all bounding boxes and saves them to a folder (+ metadata)
    given a folder with images and an annotation file (.json)
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img-dir", type=str, #default="unlabelled_images",
                        help="path to directory with unlabelled images")
    parser.add_argument("--annot-path", type=str, #default="annotations.json",
                        help="path to annotation file (.json)")
    parser.add_argument("--save-path", type=str, default="data/unlabelled_images",
                        help="a path/directory for storing the cropped images")
    
    parser.add_argument("--ignore", type=str,
                        help="(optional) path to .txt file containing labels to ignore")
    parser.add_argument("--labels", type=str,
                        help="(optional) path to .txt file containing labels to keep")
    
    args = parser.parse_args()

    if args.img_dir is None:
        print("Please enter the path to the unlabelled image directory: --img-dir <image folder>")
        exit(1)
    if args.annot_path is None:
        print("Please enter the path to the prediction file: --annot-path <json file>")
        exit(1)
        
    #if args.labels is not None:
    #    with open(args.labels) as f:
    #        lines = f.readlines()
    #        keep_list = [line.replace("\n", "") for line in lines]
    #        ignore = False
    if args.ignore is not None:
        with open(args.ignore) as f:
            lines = f.readlines()
            ignore_list = [line.replace("\n", "") for line in lines]
    else:
        ignore_list = []
    #        ignore = True
    #else:
        
        
    img_paths = get_image_paths(args.img_dir)
    if len(img_paths) == 0:
        print("Empty folder/Incorrect path: --img-dir")
        exit(1)
    
    annotations = get_annotation_data(args.annot_path)
    
    #annotations = correct_annotation_format(annotations, nested_key=args.nested_key)
    
    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)
        
    # We need metadata for remembering which bounding boxes correspond to which original annotation
    with open("data/results" + os.sep + "bbox_metadata.txt", "w") as f: 
        crop_idx = 0
        for img_path in tqdm(img_paths):
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img_filename = os.path.basename(img_path)
                annotation = annotations.get(img_filename)
                
                # If annotation for this image does not exist
                if annotation is not None:
                    for bbox_idx, bbox in enumerate(annotation):
                        pt1 = bbox["top-left"]
                        pt2 = bbox["bottom-right"]
                        label = bbox['label']
                        
                        if label in ignore_list:
                            pass
                        else:
                            cropped_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

                            index = f"{crop_idx}".rjust(5, "0")
                            filename =  f"frame{index}.jpg"
                            save_path = os.path.join(args.save_path, filename)
                            success = cv2.imwrite(save_path, cropped_img)
                            if not success:
                                raise Error()
                            else:
                                f.write(f"{os.path.basename(img_path)} {filename} {bbox_idx}\n")
                                crop_idx += 1
                else:
                    print(f"No annotation for this image: {img_path}")
        
        