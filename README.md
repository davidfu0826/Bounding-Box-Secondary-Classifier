# Bounding-Box-Secondary-Classifier

## About this repository
This repository is used as a part of a semi-supervised training pipeline for object detection. Suppose you have bounding boxes (with or without labels) for an image and you want to determine the correct label for this bouding box, then this is a tool for you!

## Short description about Semi-supervised learning and usage of this repository
Suppose you have a labeled dataset **A** and an unlabeled dataset **B**. 

1. We start by training an object detection model (e.g. YOLOv3) on **A**. 
2. After the training we perform prediction on the unlabeled dataset **B** and we save the predictions (bounding boxes and labels for each image).
3. Now, this is where **this repository** comes in! From the images we crop out all the bounding boxes we received from the predictions and classify them using a **secondary classifier model** which we train in a separate environment, the point is to **make corrections to the label/class predictions**.
4. Now, we re-train the object detection model on dataset **A** + **B**.
5. Go back to step 3.

## How to use
You need to write a script to read you prediction output files and pass it the functions.

**More update will come as this repository updates!**

## Ideas to implement
- Use all prediction accuracies (no argmax, keep while output) and clustering to improve algorithm
