import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Save final results into JSON format",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bbox-data", type=str, default="results/bbox_metadata.txt",
                        help="path to .txt file containing bounding boxes")
    parser.add_argument("--results", type=str, default="results/results.txt",
                        help="path to prediction results from secondary classifier.")
    
    args = parser.parse_args()
    
    with open(args.results) as f:
        lines = f.readlines()
        
        # 0. framename 1. secondary prediction
        predictions = [line.replace("\n", "").split(" ") for line in lines]

    with open(args.bbox_data) as f:
        lines = f.readlines()
        
        # 0. filename 1. framename 2. bbox idx
        metadata = [line.replace("\n", "").split(" ") for line in lines]
        
        
    for i, (img_filename, frame_filename, bbox_idx) in enumerate(metadata):
        
        prediction = predictions[i]
        assert prediction[0] == frame_filename
        metadata[i].append(prediction[1])
        
        
    with open("example/example_annotation.json") as f:
        annotations = json.load(f)

    # Put the new labels into the annotation files
    for img_filename, _, bbox_id, label in metadata:
        annotations[img_filename][int(bbox_id)] = label
        
    with open("results/new_annotations.json", "w") as f:
        json.dump(annotations, f)