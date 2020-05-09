# Bounding-Box-Secondary-Classifier

## About this repository
This repository is used as a part of a semi-supervised training pipeline for **object detection** or **classification**. Suppose you have **large amount of unlabelled dataset** in addition to a small labelled dataset, this repository will help you make use of that large unlabelled dataset with self-training. You can view it as an EM-algorithm.

*Note: Suppose you have decided to use semi-supervised training for object detection and you have a trained model which predicts inaccurate labels on your bounding boxes. Then this repository will help you correct the labels for each bouding box, which means that this is also a tool for you!*w

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
