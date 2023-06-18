### Surveillance System with face detection and face recognition


**Abstract** 

The rise in crime and security breaches has led to an increasing demand for effective surveillance systems.  

One such system is equipped with face recognition technology, which has gained popularity in recent years due to its ability to accurately identify individuals.  

In this system, cameras capture images of individuals in the area being monitored, and these images are analysed using machine learning algorithms to match them with known faces in a database. Steps are as follows: 

- It takes inside an image of any size 
- Then a Face detection algorithm is applied on it 
- After face being detected it’s cropped out of the image such that it’ s size equals 100x100x3 
- Then it’s passed to a Face recognition model 

The system can be used for various applications, including access control, crowd management, and criminal investigations.  

The development of surveillance systems equipped with face recognition technology has provided a valuable tool for improving security and safety in various settings. 

1. **INTRODUCTION** 

This documentation is about Surveillance System with face detection and face recognition which has 2 main functionalities which are: 

- **Face Detection** 
- **Face Recognition** 

It uses 2 algorithms to implement 

- **For Face Recognition**  
- We have used Siamese Neural Networks 
- Siamese neural networks are a type of neural network architecture that can be used for face detection and other similarity-based tasks. In the context of face detection, a Siamese neural network is typically trained on pairs of images to learn a similarity metric that can be used to determine whether two images contain the same person's face or not. 
- The Siamese architecture consists of two identical sub-networks, which are trained on pairs of images and produce feature vectors that represent each image. The two feature vectors are then compared using a distance metric, such as Euclidean distance or cosine similarity, to determine the similarity between the two images. 
- In the case of face detection, a Siamese network can be trained on pairs of images, with one image containing a known face and the other containing an unknown face. The network learns to produce feature vectors that are similar for images containing the same face and dissimilar for images containing different faces. This can be used to 

create a face recognition system that can identify a person from a database of known faces. 

- Siamese networks have been shown to be effective for face detection and face recognition tasks, particularly when dealing with small datasets or situations where there is significant variation in lighting, pose, or other factors. However, they can be computationally expensive to train and may require significant resources, such as GPUs, to achieve good performance. 
- We must pass the two images parallelly through two Siamese Neural Networks  
- And then subtract the 2 embeddings (L1Dist) 
- Pass the resultant to a binary classification algorithm 
- We have used Adam’s optimiser here as optimiser. 
- We have used Binary Cross entropy as loss function 
- **For Face detection**  
- We have used haarcascade\_frontalface\_default.xml .
2. **SOFTWARE AND HARDWARE REQUIRMENTS:** 
1. **Training and saving the model: Kaggle**  
1. We could have done both the things in colab but kaggle gives a better gpu than colab in its free version too. 
1. We used Kaggle GPU P100 
1. Kaggle is a popular online platform that provides a variety of resources for training neural networks and other machine learning models. Some of the key resources that Kaggle provides for training NNs include: 
1. Datasets: Kaggle hosts a large collection of datasets that can be used to train neural networks. These datasets cover a wide range of topics, from image recognition and natural language processing to time series analysis and recommendation systems. 
1. Kernels: Kaggle provides an online code editor called Kernels, which allows users to write, execute, and share Python code in a web browser. Kernels are pre-configured with popular machine learning libraries like TensorFlow, PyTorch, and scikit-learn, making it easy to start training NNs right away. 
2. **IDE - jupyter notebook(Google colab):** 
1. We have to use google colab as it provides free gpu to run and our gpu and memory requirements are pretty high to run the model. 
1. Google Colab, short for Google Colaboratory, is a cloud-based platform for working with Jupyter notebooks that allows users to write and run Python code in a web browser without any setup or installation. It is a free service provided by Google that offers a variety of powerful computing resources, including high-performance GPUs, to enable machine learning and data analysis tasks. 
1. Google Colab offers a range of useful features, including support for popular data science libraries like TensorFlow, Keras, PyTorch, and OpenCV. Additionally, Colab provides a collaborative environment where users can share and collaborate on their notebooks with others. This makes it a great tool for data scientists, researchers, and students who need to work with data and code in a collaborative and convenient way. 

**Hardware requirements** 

- **GPU** 
- GPU P 100 Kaggle 
- The Kaggle GPU P100 is a type of GPU accelerator offered by Kaggle for running machine learning and deep learning workloads. The P100 is a powerful GPU that can accelerate training of deep learning models, particularly those involving large amounts of data and computationally intensive operations, such as training convolutional neural networks (CNNs) on large image datasets. 
- The P100 is based on the NVIDIA Pascal architecture and offers a high level of performance, with 3584 CUDA cores, 16 GB of high-bandwidth memory (HBM2), and a memory bandwidth of up to 732 GB/s. This makes it suitable for a wide range of machine learning tasks, including image recognition, natural language processing, and reinforcement learning. 
- Kaggle provides the P100 as a cloud-based service, which means you can access it from anywhere with an internet connection. This makes it easy to spin up a P100 instance for your machine learning tasks without needing to purchase expensive hardware. However, it's important to note that using the P100 can be expensive, so it's important to optimize your code and minimize unnecessary computations to make the most efficient use of the available resources. 
- RAM  
- 16GB DDR4 is recommended 
3. **EXISTING SYSTEM/APPROACH/METHOD** 
1. **Face Detection models other than ours** 
   1. **Single Shot Detector (SSD):** This is a neural network model that uses a single pass through the network to detect faces. The SSD model has shown good performance on face detection tasks, but may not be as accurate as some of the more complex models. 
   1. **Region-based Convolutional Neural Network (R-CNN):** This is a neural network model that uses region proposal networks to identify potential face regions and then applies a CNN to each region to classify whether it contains a face or not. The R-CNN model is computationally expensive, but can achieve high accuracy on face detection tasks. 
   1. **Faster R-CNN:** This is an improved version of the R-CNN model that uses a shared CNN to compute features for potential regions and then uses region proposal networks to select the most promising regions for further classification. The Faster R-CNN model is faster and more accurate than the original R-CNN model. 
   1. **RetinaNet:** This is a neural network model that uses a novel focal loss function to address the issue of class imbalance in object detection tasks. The RetinaNet model has shown good performance on face detection tasks, particularly in cases where there is significant variation in scale or pose. 
1. **Face recognition models other than ours** 
1. **Triplet Loss Networks:** This is another type of neural network architecture that can be used for similarity-based tasks, including face recognition. Triplet loss networks are trained on triplets of images, with two images containing the same person's face and one image containing a different person's face. 

The network is trained to produce feature vectors that are similar for the two images of the same person and dissimilar for the image of the different person. 

2. **DeepFace:** This is a neural network model developed by Facebook that uses a 3D model to warp faces into a canonical pose and then applies a CNN to the warped image to extract features. The DeepFace model has achieved high accuracy on face recognition tasks and is particularly effective at handling variations in pose and lighting. 
2. **FaceNet:** This is a neural network model developed by Google that uses a CNN to directly map face images to a high-dimensional embedding space, where faces of the same person are close together and faces of different people are far apart. The FaceNet model has achieved state-of-the-art performance on a wide range of face recognition tasks. 
2. **VGGFace:** This is a neural network model that uses a CNN to extract features from face images and then applies a fully connected layer to classify the face images into identities. The VGGFace model has achieved high accuracy on face recognition tasks and is particularly effective at handling variations in expression and occlusion. 

**Why we chose Haar cascaded?** 

- In terms of computational requirements, the Haar Cascades model is generally the least computationally intensive, followed by the Single Shot Detector (SSD) model. 
- The Region-based Convolutional Neural Network (R-CNN) and Faster R- CNN models are more computationally expensive, as they require a region proposal step that can be time-consuming. These models also require significant amounts of memory, which can be a limiting factor on some hardware platforms. 
- The RetinaNet model is also computationally expensive, but has been shown to be more efficient than other state-of-the-art models, such as the Faster R-CNN model, due to its use of a novel focal loss function. 

**Why we choose Siamese Neural networks?** 

Siamese Neural Networks are suitable for similarity-based tasks like face recognition, signature verification, and text matching. They can be trained on pairs of data samples to learn a similarity metric, making them useful in cases with limited labeled data. The network can handle varying data sizes and shapes, extracting meaningful features from non-uniform data. Siamese networks can recognize small differences between data samples, making them useful for subtle tasks like signature verification or text matching.** 

4. **Proposed/Developed Model**  
1. **Design** 

![](Aspose.Words.450f599c-c908-413b-9abb-ba2589a5d7d5.003.png)

- Here firstly we get users image then it’s fed to Face Detection algorithm  
- Then from the face detection algorithm we crop the face of size 100 pixel x 100 pixel 
- Then that is passed to Face recognition algorithm where it matches it to some 50 samples and then give the average of like how much percentage it’s matching to it . 
2. **Architectural Design of Siamese neural networks** 

![](Aspose.Words.450f599c-c908-413b-9abb-ba2589a5d7d5.004.jpeg)

3. **Module Wise Description** 
1. Face Detection 
1. Image is cropped and is of size 100x100 pixel 
1. Face Recognition 

1\.  Setup 

1. Install Dependencies 
1. Import Dependencies 
1. Set GPU Growth 
   1. By default, Tensorflow allocates all available GPU memory for the training process, which can cause the system to run out of memory if the model is large or the GPU has limited memory.Setting the memory growth option to True ensures that Tensorflow will allocate GPU memory dynamically, rather than allocating all of it upfront. 
1. Create Folder Structures 
1. Positive 
   1. Which will contain image which if compared to anchor must give 1 as label 
1. Negative 
   1. Which will contain image which if compared to anchor must give 0 as label 
1. Anchor 
   1. This is the image which we have to compare to 
2. Collect Positives and Anchors 
   1. Untar Labelled Faces in the Wild Dataset 
   1. Collect Positive and Anchor Classes 
2. Load and Preprocess Images 
1. Get Image Directories 
1. Preprocessing - Scale and Resize 
1. Scaling 
1. Machine learning algorithms and Deep learning models require the input data to be normalized, which means that the values are scaled to a specific range, often [0, 1] or [-1, 1].
2. Normalizing the image to the range [0, 1] increases the memory usage by a factor of 4.
2. Scaling the image to the range [0, 1] before storing it, we can avoid the need to divide each pixel value by 255 during training,
2. Resize 

1\.  Resizing is important as for training and testing the data which should be fed to the model should be similar 

3. Create Labelled Dataset 
3. Build Train and Test Partition 
4. Model Engineering 
1. Build Embedding Layer 
1. Build Distance Layer 
1. ![](Aspose.Words.450f599c-c908-413b-9abb-ba2589a5d7d5.005.png)
1. Here input embedding and validation embedding 

are the images passed through the Siamese model 

and these are a column vector 

3. Make Siamese Model 
5. Training 
1. Setup Loss and Optimizer 

i.  We have used Binary cross entropy(loss function) 

and Adams optimiser 

2. Establish Checkpoints 
   1. If model fails it will help 
2. Build Train Step Function 
2. Build Training Loop 
2. Train the model 
6. Evaluate Model 
1. Here we have divided the training data in batches of 16 each 
1. We have evaluated the model on the basis of precision and 

recall 

1. Precision: Precision measures how accurate a 

model's positive predictions are. It is calculated as 

the number of true positive predictions divided by 

the total number of positive predictions made by 

the model. Precision can be expressed as: 

1\.  Precision = true positives / (true positives + 

false positives) 

2. Recall: Recall measures how well a model can 

identify all the positive instances. It is calculated as 

the number of true positive predictions divided by the total number of actual positive instances in the dataset. Recall can be expressed as: 

1\.  Recall = true positives / (true positives + false negatives) 

7\.  Save Model 

a.  We saved the model by the name of siamesemodelv2.h5

d.  Implementation 

1. Till model saving 
   1. [https://www.kaggle.com/code/sakshamverma778/softcomputingm odelrecognition/notebook?scriptVersionId=125772047 ](https://www.kaggle.com/code/sakshamverma778/softcomputingmodelrecognition/notebook?scriptVersionId=125772047) 
1. Implementation 
1. [https://colab.research.google.com/drive/14T2YZk1xgAzsoTtTHdn65 sWeHQBEWZII?usp=sharing ](https://colab.research.google.com/drive/14T2YZk1xgAzsoTtTHdn65sWeHQBEWZII?usp=sharing) 
1. For testing  
1. We should have a folder named “verification\_images” which will contain images of the person for whom facial recognition will work 
1. It will take the cropped image from the camera and then will compare it to all these 50 images in “verification\_images” folder and for each image it will give how much percent it’s matching after that it will be passed to a classification algorithm which will make it as positive if >0.5 else will mark it as 0 then an average is taken for all the 0/1 values we have. Which will tell how much the image resembles with the person whose 5o images are there in the folder “verification\_images”. 
5. Results and Discussion 
- Face Recognition 

![](Aspose.Words.450f599c-c908-413b-9abb-ba2589a5d7d5.006.png)

Here 

- Face Detection and Face cropping 

![](Aspose.Words.450f599c-c908-413b-9abb-ba2589a5d7d5.007.jpeg)

**IMAGE CROPPING** 

**FACE RECOGNITION** 

Here we have used Siamese Neural Networks as they are suitable for similarity-based tasks like face recognition, signature verification, and text matching. They can be trained on pairs of data samples to learn a similarity metric, making them useful in cases with limited labeled data. The network can handle varying data sizes and shapes, extracting meaningful features from non-uniform data. Siamese networks can recognize small differences between data samples, making them useful for subtle tasks like signature verification or text matching 

**Face Cropping** 

Cropping is important as for training and testing the data which should be fed to the model should be similar but the face size may differ depending on the distance of the person from the camera  

**FACE DETECTION** 

We have used  harcascade 

- In terms of computational requirements, the Haar Cascades model is generally the least computationally intensive, followed by the Single Shot Detector (SSD) model. 
- The Region-based Convolutional Neural Network (R-CNN) and Faster R-CNN models are more computationally expensive, as they require a region proposal step that can be time-consuming. These models also require significant amounts of memory, which can be a limiting factor on some hardware platforms. 
- The RetinaNet model is also computationally expensive, but has been shown to be more efficient than other state-of-the-art models, such as the Faster R-CNN model, due to its use of a novel focal loss function. 
6. **Conclusion (Mention limitations in your project and how it can be enhanced) For Face Recognition**  

We have used Siamese Neural Networks 

1. Siamese neural networks are a type of neural network architecture that can be used for face detection and other similarity-based tasks. In the context of face detection, a Siamese neural network is typically trained on pairs of images to learn a similarity metric that can be used to determine whether two images contain the same person's face or not. 
1. The Siamese architecture consists of two identical sub-networks, which are trained on pairs of images and produce feature vectors that represent each image. The two feature vectors are then compared using a distance metric, such as Euclidean distance or cosine similarity, to determine the similarity between the two images. 
1. In the case of face detection, a Siamese network can be trained on pairs of images, with one image containing a known face and the other containing an unknown face. The network learns to produce feature vectors that are similar for images containing the same face and dissimilar for images containing different faces. This can be used to create a face recognition system that can identify a person from a database of known faces. 
1. Siamese networks have been shown to be effective for face detection and face recognition tasks, particularly when dealing with small datasets or situations where there is significant variation in lighting, pose, or other 

factors. However, they can be computationally expensive to train and may require significant resources, such as GPUs, to achieve good performance. 

**For Face detection**  

- We have used haarcascade\_frontalface\_default.xml .
- In terms of computational requirements, the Haar Cascades model is generally the least computationally intensive, followed by the Single Shot Detector (SSD) model. 
- The Region-based Convolutional Neural Network (R-CNN) and Faster R- CNN models are more computationally expensive, as they require a region proposal step that can be time-consuming. These models also require significant amounts of memory, which can be a limiting factor on some hardware platforms. 
- The RetinaNet model is also computationally expensive, but has been shown to be more efficient than other state-of-the-art models, such as the Faster R-CNN model, due to its use of a novel focal loss function. 

**Mention limitations in your project and how it can be enhanced** 

1. This project has only being designed for a single person face detection and in future model should be trained on more people which will help to predict more number of peoples 
1. We haven’t integrated it to any surveillance device it’s currently under prototyping stage for that purpose 
7. **REFERENCES:** 
- [https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf ](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [https://www.tensorflow.org/ ](https://www.tensorflow.org/) 
- [https://www.youtube.com/watch?v=YjWh7QvVH60 ](https://www.youtube.com/watch?v=YjWh7QvVH60) 
- [https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade- opencv/#:~:text=What%20are%20Haar%20Cascades%3F,%2C%20buildings%2C%20fruits %2C%20etc.](https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/#:~:text=What%20are%20Haar%20Cascades%3F,%2C%20buildings%2C%20fruits%2C%20etc) 
8. **APPENDIX: SAMPLE CODE** 
1. Till model saving** 
   1. [https://www.kaggle.com/code/sakshamverma778/softcomputingm odelrecognition/notebook?scriptVersionId=125772047 ](https://www.kaggle.com/code/sakshamverma778/softcomputingmodelrecognition/notebook?scriptVersionId=125772047)
1. Implementation [https://colab.research.google.com/drive/14T2YZk1xgAzsoTtTHdn65 sWeHQBEWZII?usp=sharing ](https://colab.research.google.com/drive/14T2YZk1xgAzsoTtTHdn65sWeHQBEWZII?usp=sharing) 
12 
