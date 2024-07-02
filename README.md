# Creating the README.md file content in multiple parts to avoid incomplete input error

part1 = """
# 📸 Surveillance System with Face Detection and Face Recognition
---

## 📑 Table of Contents
1. [Abstract](#0-abstract)
2. [Introduction](#1-introduction)
3. [Hardware / Software Requirements](#2-hardware--software-requirements)
4. [Existing System/Approach/Method](#3-existing-systemapproachmethod)
5. [Proposed/Developed Model](#4-proposeddeveloped-model)
6. [Results and Discussion](#5-results-and-discussion)
7. [Conclusion](#6-conclusion)
8. [References](#7-references)
9. [Appendix: Sample Code](#8-appendix-sample-code)

---

## 0. Abstract

The rise in crime and security breaches has led to an increasing demand for effective surveillance systems. One such system is equipped with face recognition technology, which has gained popularity in recent years due to its ability to accurately identify individuals. In this system, cameras capture images of individuals in the area being monitored, and these images are analyzed using machine learning algorithms to match them with known faces in a database.

### Steps:
- 📷 It takes inside an image of any size.
- 🕵️ Then a face detection algorithm is applied to it.
- ✂️ After the face is detected, it is cropped out of the image such that its size equals 100x100x3.
- 🧠 Then it’s passed to a face recognition model.

The system can be used for various applications, including access control, crowd management, and criminal investigations. The development of surveillance systems equipped with face recognition technology has provided a valuable tool for improving security and safety in various settings.

---

## 1. Introduction

This documentation is about a Surveillance System with face detection and face recognition which has two main functionalities:
- 🔍 Face Detection
- 🧠 Face Recognition

### Algorithms used:
- **For Face Recognition:**
  - **Siamese Neural Networks**
    - Siamese neural networks are a type of neural network architecture used for face detection and other similarity-based tasks. In the context of face detection, a Siamese neural network is typically trained on pairs of images to learn a similarity metric that can be used to determine whether two images contain the same person's face or not.
    - The Siamese architecture consists of two identical sub-networks, which are trained on pairs of images and produce feature vectors that represent each image. The two feature vectors are then compared using a distance metric, such as Euclidean distance or cosine similarity, to determine the similarity between the two images.
    - In the case of face detection, a Siamese network can be trained on pairs of images, with one image containing a known face and the other containing an unknown face. The network learns to produce feature vectors that are similar for images containing the same face and dissimilar for images containing different faces. This can be used to create a face recognition system that can identify a person from a database of known faces.
    - Siamese networks have been shown to be effective for face detection and face recognition tasks, particularly when dealing with small datasets or situations where there is significant variation in lighting, pose, or other factors. However, they can be computationally expensive to train and may require significant resources, such as GPUs, to achieve good performance.
  - 📊 Images are passed parallelly through two Siamese Neural Networks, and the embeddings are subtracted (L1Dist), followed by a binary classification algorithm. Adam’s optimizer and Binary Cross Entropy are used as the loss function.

- **For Face Detection:**
  - **Haar Cascade Classifier** (haarcascade_frontalface_default.xml)

---

## 2. Hardware / Software Requirements

### Training and Saving the Model: Kaggle
- We used Kaggle GPU P100 for training.
- Kaggle provides a variety of resources for training neural networks and other machine learning models:
  - **Datasets:** A large collection of datasets covering various topics.
  - **Kernels:** An online code editor for writing, executing, and sharing Python code with pre-configured machine learning libraries.

### IDE: Jupyter Notebook (Google Colab)
- Google Colab provides free GPU and a collaborative environment for working with Jupyter notebooks.

### Hardware Requirements:
- **GPU:** Kaggle GPU P100
  - The Kaggle GPU P100 is a powerful GPU based on the NVIDIA Pascal architecture, suitable for a wide range of machine learning tasks.
- **RAM:** 16GB DDR4 is recommended.

---

## 3. Existing System/Approach/Method

### Face Detection Models
- **Single Shot Detector (SSD):** A neural network model that uses a single pass to detect faces.
- **Region-based Convolutional Neural Network (R-CNN):** Uses region proposal networks and CNNs for face detection, high accuracy but computationally expensive.
- **Faster R-CNN:** An improved version of R-CNN with faster and more accurate performance.
- **RetinaNet:** Uses a novel focal loss function, good for handling variations in scale or pose.

### Face Recognition Models
- **Triplet Loss Networks:** Trained on triplets of images for similarity-based tasks.
- **DeepFace:** Developed by Facebook, uses a 3D model to warp faces and a CNN for feature extraction.
- **FaceNet:** Developed by Google, maps face images to a high-dimensional embedding space.
- **VGGFace:** Uses a CNN for feature extraction and a fully connected layer for classification.

### Why Haar Cascades?
- Least computationally intensive among the listed models.
- Suitable for hardware with limited memory and computational resources.

### Why Siamese Neural Networks?
- Suitable for similarity-based tasks like face recognition.
- Effective with limited labeled data and non-uniform data.
- Recognizes small differences between data samples.
"""

part2 = """
---

## 4. Proposed/Developed Model

### Design
- The user's image is fed to a face detection algorithm.
- The detected face is cropped to 100x100 pixels.
- The cropped image is passed to a face recognition algorithm, matching it against 50 samples and averaging the match percentage.

### Architectural Design of Siamese Neural Networks

### Module Wise Description
1. **Face Detection**
2. **Image Cropping:** Image is cropped to 100x100 pixels.
3. **Face Recognition**
    - **Setup:**
        - Install Dependencies
        - Import Dependencies
        - Set GPU Growth: Ensures dynamic GPU memory allocation.
        - Create Folder Structures: Positive, Negative, and Anchor.
        - Collect Positives and Anchors: Using the Labeled Faces in the Wild Dataset.
        - Load and Preprocess Images: Scaling and resizing.
    - **Model Engineering:**
        - Build Embedding Layer
        - Build Distance Layer
        - Make Siamese Model
    - **Training:**
        - Setup Loss and Optimizer: Binary Cross Entropy and Adam's optimizer.
        - Establish Checkpoints
        - Build Train Step Function
        - Build Training Loop
        - Train the model
    - **Evaluate Model:**
        - Evaluated based on precision and recall.
    - **Save Model:** Saved as siamesemodelv2.h5

### Implementation
- **Till model saving:** [Kaggle Notebook](https://www.kaggle.com/code/sakshamverma778/softcomputingmodelrecognition/notebook?scriptVersionId=125772047)
- **Implementation:** [Google Colab](https://colab.research.google.com/drive/14T2YZk1xgAzsoTtTHdn65sWeHQBEWZII?usp=sharing)
    - Testing requires a folder named “verification_images” containing images of the person for whom facial recognition will work.

---

## 5. Results and Discussion

### Face Recognition

### Face Detection and Face Cropping

### Face Recognition
- Used Siamese Neural Networks for similarity-based tasks like face recognition.

### Face Cropping
- Important for training and testing, ensuring similar data input.

### Face Detection
- Haar Cascades: Least computationally intensive, followed by SSD, R-CNN, and Faster R-CNN. RetinaNet is computationally expensive but efficient.

---

## 6. Conclusion

### For Face Recognition
- Used Siamese Neural Networks for effective face detection and recognition.
- Siamese networks are trained on pairs of images, producing feature vectors compared using a distance metric.

### For Face Detection
- Used haarcascade_frontalface_default.xml.
- Haar Cascades are computationally less intensive compared to other models.

### Limitations and Enhancements
- Currently designed for single-person face detection. Future training should include more people.
- The project is in the prototyping stage and not yet integrated with any surveillance device.

---

## 7. References
- [Siamese Neural Networks for One-Shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [TensorFlow](https://www.tensorflow.org/)
- [Face Recognition with OpenCV](https://www.youtube.com/watch?v=YjWh7QvVH60)
- [Object Detection using Haar Cascade](https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/#:~:text=What%20are%20Haar%20Cascades%3F,%2C%20buildings
