import argparse
import os
import glob
from random import shuffle
from collections import Counter
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder

from utils.models import get_model
from utils.data import get_train_transforms, get_test_transforms, CustomImageDataset, undersample, oversample

def get_dataset(image_folder: str, img_size: str, self_training: bool = False):
    """Returns DataLoaders, class decoder and weights (for balancing dataset)
    which can be used for PyTorch model training. 
    
    Args:
        image_folder:  Path to directory with <class>/<image> structure
        img_size:      Size of images in training dataset
        self_training: Turn on training with unlabelled dataset
    """

    primary_img_paths = glob.glob(image_folder + os.sep + "*/*.jpg")
    primary_img_paths += glob.glob(image_folder + os.sep + "*/*.png")
    
    #primary_img_paths = undersample(primary_img_paths)
    
    SIZE = len(primary_img_paths)
    shuffle(primary_img_paths)
        
    TRAIN = int(SIZE*TRAIN_RATIO)
    TEST = SIZE - TRAIN
 
    if self_training:
        print("Using predictions on unlabelled data in train set!".rjust(70, "#").ljust(90, "#"))
        secondary_img_path = glob.glob("data/secondary_dataset" + os.sep + "*/*.jpg")
        shuffle(secondary_img_path)

        train_img_paths = primary_img_paths[:TRAIN] + secondary_img_path
    else:
        train_img_paths = primary_img_paths[:TRAIN]
        
    test_img_paths = primary_img_paths[TRAIN:]
    TRAIN = len(train_img_paths)  # For display purpose
    
    if self_training:
        TRAIN += len(secondary_img_path) # For display purpose
    
    train_dataset = CustomImageDataset(train_img_paths, get_train_transforms(img_size))
    test_dataset = CustomImageDataset(test_img_paths, get_test_transforms(img_size))
    class_to_idx = train_dataset.class_to_idx
    
    # Create DataLoader for training
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    weights = get_class_weights(train_img_paths, class_to_idx) # For balancing dataset using inverse-frequency
        
    print(f"Number of classes {NUM_CLASSES}, Train size: {TRAIN} images, Test size: {TEST} images, Batch size: {BATCH_SIZE}, Image size: {img_size}x{img_size}")
    return train_dataloader, test_dataloader, class_to_idx, weights

def get_class_weights(img_paths: List[str], class_to_idx: Dict[str, int]):
    """Helper function for calculating the weights
    for each class. This can be used for balancing imbalanced datasets.
    
    Args:
        img_paths:    List with paths to samples
        class_to_idx: Class encoder
    """
    labels = list()
    for img_path in img_paths:
        label = os.path.basename(os.path.dirname(img_path))
        labels.append(class_to_idx[label]) 
    counts = Counter(labels)
    counts = np.array(sorted(counts.items()))[:,1]
    return counts.max()/counts
    

def get_training_stuff(model: nn.Module, weights: torch.Tensor = None):
    """Returns optimizer and criterion for PyTorch training.
    
    Args:
        model:   PyTorch model object
        weights: PyTorch tensor containing inverse frequencies
    """
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(weight=weights)
    return optimizer, criterion


def show_confusion_matrix(matrix: List[List], labels: List[str]):
    """Display a nice confusion matrix given
    the confusion matrix in a 2D list + list of labels (decoder)
    
    Args:
        matrix: 2D array containing the values to display (confusion matrix)
        labels: Array containing the labels (indexed by corresponding label idx)
    """
    fig, ax = plt.subplots()
    fig.set_figheight(15)
    fig.set_figwidth(15)

    min_val, max_val = 0, len(labels)

    for i in range(max_val):
        for j in range(max_val):
            c = matrix[i][j]
            ax.text(i, j, str(int(c)), va='center', ha='center')

    ax.matshow(matrix, cmap=plt.cm.Blues)

    # Set number of ticks for x-axis
    ax.set_xticks(np.arange(max_val))
    # Set ticks labels for x-axis
    ax.set_xticklabels(labels, rotation='vertical', fontsize=16)

    # Set number of ticks for x-axis
    ax.set_yticks(np.arange(max_val))
    # Set ticks labels for x-axis
    ax.set_yticklabels(labels, rotation='horizontal', fontsize=16)
                    
    #ax.set_xlim(min_val, max_val)
    ax.set_ylim(max_val - 0.5, min_val - 0.5)
    plt.show()
    
def display_missclassified(class_to_idx: Dict[str,int], 
                           targets: List[int], 
                           predictions: List[int], 
                           images: List[np.ndarray], 
                           gridsize: Tuple[int] = (4,4)):
    """Display a grid with missclassified samples from test set.
    
    Args:
        class_to_idx: Class to idx encoder
        targets:      List containing all ground truths
        predictions:  List containing all predictions
        images:       List containing image arrays
        gridsize:     Tuple describing the final image grid
    """
    fig = plt.figure()
    plot_counter = 1
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    idx_to_class = {i:label for i, label in enumerate(class_to_idx)}
    for i in range(len(targets)):
        if plot_counter > gridsize[0]*gridsize[1]:
            break
        
        image = images[i].transpose(1, 2, 0)
        image = ((image * std) + mean) * 255
        image = image.astype("uint8")
    
        image = cv2.resize(image, (128, 128))
        image = cv2.putText(image, idx_to_class[predictions[i]], (0,20), 3, 0.4, (0,0,255), 1)
        if predictions[i] == targets[i]:
            pass
        else:
            ax = fig.add_subplot(gridsize[0], gridsize[1], plot_counter)
            ax.imshow(image)
            plot_counter += 1
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Trains a secondary model",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, default="labelled_images",
                        help="path to directory with dataset according to: <dataset-dir>/<class>/<img>")
    
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size used during training")
    
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="ratio between train and test")

    parser.add_argument("--img-size", type=int, default=32,
                        help="image size used for training")
    
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs (1 epoch = 1 full dataset iteration)")
    
    parser.add_argument("--weights", type=str, default="weights",
                        help="weights file directory")
    
    parser.add_argument("--resume", action="store_true",
                        help="resume training")
    
    parser.add_argument("--self-training", action="store_true",
                        help="semi-supervised training (requires labelled unlabelled dataset)")
    
    args = parser.parse_args()
    
    TRAIN_RATIO = args.train_ratio
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = 19
    IMG_SIZE = args.img_size
    epochs = args.epochs
    WEIGHTS_DIR = args.weights
    RESUME = args.resume
    
    if RESUME:
        state_dict = torch.load(os.path.join(WEIGHTS_DIR, "last.pt"))
        
        train_dataloader = state_dict["train_dataloader"]
        test_dataloader = state_dict["test_dataloader"]
        class_to_idx = state_dict["class_to_idx"]
    
    else:   
        image_folder = args.dataset_dir
        if image_folder is None:
            print("Please enter the path to the labelled dataset by '--dataset-dir'")
            exit(1)
        else:
            if not os.path.isdir(image_folder):
                print("Please enter a correct path to the dataset by '--dataset-dir'")
                exit(1)        
            else:
                pass
            
        train_dataloader, test_dataloader, class_to_idx, weights = get_dataset(image_folder, IMG_SIZE, self_training = args.self_training)
    
    # Using gpu or not
    CUDA = "cuda" if torch.cuda.is_available() else "cpu"
    if CUDA == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    model = get_model(NUM_CLASSES)
    model.to(CUDA)
    print(list(class_to_idx.keys()))
    
    if RESUME:
        optimizer, criterion = get_training_stuff(model)
    else:
        weights = torch.Tensor(weights).to(CUDA)
        optimizer, criterion = get_training_stuff(model, weights=weights)
    
    if RESUME:
        start_epoch = state_dict["epoch"]
        optimizer_state_dict = state_dict["optimizer_state_dict"]
        best_test_f1 = state_dict["best_test_f1"]
        
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    else:
        best_test_f1 = 0
        start_epoch = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0
        t = tqdm(train_dataloader)
        for i, (X, y) in enumerate(t):
 
            X = X.to(CUDA)
            y = y.to(CUDA)

            optimizer.zero_grad()

            preds = model(X)

            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().detach()
            t.set_description(f"{epoch+1}/{epochs} Train: {round(float(running_loss)/(i+1), 4)}")

        model.eval()
        predictions = list() # For display purpose
        targets = list() # For display purpose
        if epoch+1 == epochs:
            images = list() # For display purpose
        running_loss = 0
        t = tqdm(test_dataloader)
        for i, (X, y) in enumerate(t):
            X = X.to(CUDA)
            y = y.to(CUDA)

            preds = model(X)
            predictions += list(preds.argmax(axis=1).cpu().detach().numpy())
            targets += list(np.array(y.cpu()))
            if epoch+1 == epochs:
                images += list(np.array(X.cpu()))

            loss = criterion(preds, y)

            running_loss += loss.cpu().detach()
            t.set_description(f"Test: {round(float(running_loss/(i+1)), 4)}")
            
        acc = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average="macro", labels=np.unique(predictions))
        recall = recall_score(targets, predictions, average="macro", labels=np.unique(predictions))
        precision = precision_score(targets, predictions, average="macro", labels=np.unique(predictions))
        print(f"Test: Acc: {str(acc)[:5]}, F1: {str(f1)[:5]}, Recall: {str(recall)[:5]}, Precision: {str(precision)[:5]}\n")

        if f1 > best_test_f1:
            best_test_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_f1': best_test_f1,
                'train_dataloader': train_dataloader,
                'test_dataloader': test_dataloader,
                'class_to_idx': class_to_idx
            }, os.path.join(WEIGHTS_DIR, "best.pt"))
            
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_f1': best_test_f1,
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'class_to_idx': class_to_idx,
    }, os.path.join(WEIGHTS_DIR, "last.pt"))
    
    display_missclassified(class_to_idx, targets, predictions, images, gridsize=(4,4))
    show_confusion_matrix(confusion_matrix(targets, predictions), list(class_to_idx.keys()))
    