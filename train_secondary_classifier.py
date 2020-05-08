import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder

from utils.models import get_model
from utils.data import get_transforms

def get_dataset(image_folder, img_size):
    dataset = ImageFolder(image_folder, get_transforms(img_size))
    SIZE = len(dataset)
    TRAIN = int(SIZE*TRAIN_RATIO)
    TEST = SIZE - TRAIN
    train_dataset, test_dataset = random_split(dataset, [TRAIN, TEST])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    class_to_idx = dataset.class_to_idx
    return train_dataloader, test_dataloader, class_to_idx

def get_training_stuff(model):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def show_confusion_matrix(matrix, labels):
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
    #ax.set_xticks(np.arange(max_val))
    #ax.set_yticks(np.arange(max_val))
    #ax.grid()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", type=str, #default="unlabelled_images",
                        help="path to directory with dataset according to: <dataset-dir>/<class>/<img>")
    
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="batch size used during training")
    
    parser.add_argument("-t", "--train-ratio", type=float, default=0.8,
                        help="ratio between train and test")

    parser.add_argument("-i", "--img-size", type=int, default=32,
                        help="image size used for training")
    
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of training epochs (1 epoch = 1 full dataset iteration)")
    
    parser.add_argument("-w", "--weights", type=str, default="weights",
                        help="weights file directory")
    
    parser.add_argument("--resume", action="store_true",
                        help="resume training")
    
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
            print("Please enter the path to the dataset by '--dataset-dir'")
            exit(1)
        else:
            if not os.path.isdir(image_folder):
                print("Please enter a correct path to the dataset by '--dataset-dir'")
                exit(1)        
            else:
                pass
            
        train_dataloader, test_dataloader, class_to_idx = get_dataset(image_folder, IMG_SIZE)
    
    # Using gpu or not
    CUDA = "cuda" if torch.cuda.is_available() else "cpu"
    if CUDA == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    model = get_model(NUM_CLASSES)
    model.to(CUDA)
    print(f"NUM_CLASSES {NUM_CLASSES}, TRAIN {len(train_dataloader)}, TEST {len(test_dataloader)}")
    
    optimizer, criterion = get_training_stuff(model)
    
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
        predictions = list()
        targets = list()
        running_loss = 0
        t = tqdm(test_dataloader)
        for i, (X, y) in enumerate(t):
            X = X.to(CUDA)
            y = y.to(CUDA)

            preds = model(X)
            predictions += list(np.array(preds.argmax(axis=1).cpu()))
            targets += list(np.array(y.cpu()))

            loss = criterion(preds, y)

            running_loss += loss.cpu().detach()
            t.set_description(f"Test: {round(float(running_loss/(i+1)), 4)}")
            
        acc = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average="macro", labels=np.unique(predictions))
        recall = recall_score(targets, predictions, average="macro", labels=np.unique(predictions))
        precision = precision_score(targets, predictions, average="macro", labels=np.unique(predictions))
        print(f"Test: Acc: {str(acc)[:5]}, F1: {str(f1)[:5]}, Recall: {str(recall)[:5]}, Precision: {str(precision)[:5]}")

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
    
    show_confusion_matrix(confusion_matrix(targets, predictions), list(class_to_idx.keys()))
    #plt.imshow(confusion_matrix(targets, predictions, normalize="true"))
    #plt.show()