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
    img_paths = glob.glob(path_to_imgs + os.sep + "*.jpg") + glob.glob(path_to_imgs + os.sep + "*.png")
    return img_paths

def get_label_paths(path_to_txts: str) -> List[str]:
    """Return all paths to labels given path to the directory
    
    Args: 
        path_to_imgs: Path to image directory
    """
    label_paths = glob.glob(os.path.join(path_to_txts, "*.txt"))
    return label_paths

def json2dict(path_to_annot: str):
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
    parser.add_argument("--label-dir", type=str,
                        help="path to labels directory")
    parser.add_argument("--save-path", type=str, default="data/unlabelled_images",
                        help="a path/directory for storing the cropped images")
    parser.add_argument("--labelled-dataset", action='store_true',
                        help="Save as <save_path>/<class_name>/*.jpg")
    parser.add_argument("--unlabelled-dataset", action='store_true',
                        help="Save as <save_path>/*.jpg")
    parser.add_argument("--names-path", type=str,
                        help="File containing name of each class.")
    
    parser.add_argument("--ignore", type=str,
                        help="(optional) path to .txt file containing labels to ignore")
    
    parser.add_argument("--keep-only", type=str,
                        help="(optional) keep only this label")
    
    
    args = parser.parse_args()

    if args.img_dir is None:
        print("Please enter the path to the unlabelled image directory: --img-dir <image folder>")
        exit(1)
    if args.annot_path is None and args.label_dir is None:
        print("Please enter the path to the prediction file (--annot-path <json file>) or the labels directory (--label-dir <label folder>).")
        exit(1)
        
    if args.labelled_dataset and args.unlabelled_dataset:
        print("Please enter choose either --labelled-dataset or --unlabelled-dataset.")
        exit(1)
    if not args.labelled_dataset and not args.unlabelled_dataset:
        print("Ambiguity, please enter choose one of --labelled-dataset or --unlabelled-dataset.")
        exit(1)
    if args.keep_only is not None:
        if args.ignore is not None:
            print("Ambiguity")
            exit(1)
   
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
    
    if args.annot_path is not None and args.label_dir is not None:
        print("Ambiguity, both --annot-path and --labels-dir are defined. Use only one of them.")
        exit(1)
        
    if args.label_dir is None: 
        annotations = json2dict(args.annot_path)
    else:
        annot_paths = get_label_paths(args.label_dir)
        
        if args.names_path is None and args.labelled_dataset:
            print("Please enter path to names file (--names-path <names of each class>).")
            exit(1)
   
        if args.labelled_dataset:
            with open(args.names_path) as f:
                idx2name = f.readlines()
            idx2name = {idx:name.replace("\n", "") for idx, name in enumerate(idx2name) if name != ""}
        
    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
        
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if args.labelled_dataset:
        for idx in idx2name:
            Path(os.path.join(args.save_path, idx2name[idx])).mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)
        
    # We need metadata for remembering which bounding boxes correspond to which original annotation
    with open("data/results" + os.sep + f"{os.path.basename(args.img_dir)}_bbox_metadata.txt", "w") as metadata: 
        crop_idx = 0
        for idx, img_path in tqdm(enumerate(img_paths)):
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img_filename = os.path.basename(img_path)
                
                if annot_paths is not None:
                    # Label and image have same filename
                    assert os.path.basename(annot_paths[idx]).replace(".txt", ".jpg") == img_filename
                    
                    with open(annot_paths[idx]) as f:
                        img_h = img.shape[0]
                        img_w = img.shape[1]
                        bboxes = f.readlines() #class_id center_x, center_y, bbox_w, bbox_h
                        bboxes = [bbox.replace("\n", "").split() for bbox in bboxes if bbox != "\n"]
                        bboxes = [bbox for bbox in bboxes if len(bbox) != 0]
                        for bbox_idx, bbox in enumerate(bboxes):
                            
                            label_idx = int(bbox[0])
                            bbox_cx =   float(bbox[1]) * img_w
                            bbox_cy =   float(bbox[2]) * img_h
                            bbox_w =    float(bbox[3]) * img_w
                            bbox_h =    float(bbox[4]) * img_h
                            tlx = int(bbox_cx-bbox_w/2)
                            tlx = tlx if tlx >= 0 else 0
                            tly = int(bbox_cy-bbox_h/2)
                            tly = tly if tly >= 0 else 0
                            brx = int(bbox_cx+bbox_w/2)
                            brx = brx if brx >= 0 else 0
                            bry = int(bbox_cy+bbox_h/2)
                            bry = bry if bry >= 0 else 0
                            
                            try:
                                name = idx2name.get(label_idx)
                            except:
                                name = None
                            
                            bboxes[bbox_idx] = {"label":        name,
                                                "top-left":     (tlx, tly),
                                                "bottom-right": (brx, bry)}
                        for bbox_idx, bbox in enumerate(bboxes):
         
                            pt1 = bbox["top-left"]
                            pt2 = bbox["bottom-right"]
                            label = bbox['label']
                            
                            if label in ignore_list:
                                pass
                            else:
                                if args.keep_only is not None:
                                    if label != args.keep_only:
                                        break
                                
                                cropped_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

                                index = f"{crop_idx}".rjust(5, "0")
                                
                                filename =  f"frame{index}.jpg"
                                if args.labelled_dataset:
              
                                    filename = os.path.join(label, f"frame{index}.jpg")
                                    
                                save_path = os.path.join(args.save_path, filename)
                                try:
                                    success = cv2.imwrite(save_path, cropped_img)
                                except:
                                    print(f"{annot_paths[idx]} contains invalid data.")
                                if not success:
                                    raise Exception()
                                else:
                                    metadata.write(f"{os.path.basename(img_path)} {filename} {bbox_idx}\n")
                                    crop_idx += 1
                else:  
                    annotation = annotations.get(img_filename)
                
                """
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
                                raise Exception()
                            else:
                                f.write(f"{os.path.basename(img_path)} {filename} {bbox_idx}\n")
                                crop_idx += 1
                else:
                    print(f"No annotation for this image: {img_path}")"""
        
        