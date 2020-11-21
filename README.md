[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"

# Dog Classification Project

## Project Source

This repository contains project#2 related to Udacity's [Deep Learning Nanodegree program](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification). 

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

## Datasets

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
2. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz). 

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
	
2. Make sure you have installed the necessary Python packages as the following.

* PyTorch and Torchvision (__Linux__ or __Mac__ )

	```	
		conda install pytorch torchvision -c pytorch
	```
* PyTorch and Torchvision (__Windows__)

	```	
		conda install pytorch -c pytorch
		pip install torchvision
	```

* Required pip packages (including OpenCV):

	```	
		pip install -r requirements.txt
	```

## My Answer Method 

### Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

I decided to make the architecture is simple and try to play with the hyperparameters to enhance the accuracy.


CNN Architecture:


	(conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

	(conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	
	(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	
	(fc1): Linear(in_features=100352, out_features=500, bias=True)
	
	(fc2): Linear(in_features=500, out_features=133, bias=True)
	
	(dropout): Dropout(p=0.2, inplace=False)


The Final Test Loss = 3.81 and Test Accuracy =  16%


### Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
In this step, as we use pre-trained models, we should trade-off between the Accuracy and Execution Time of these models. Therefore, I tried two pre-trained models: VGG16 and ResNet for just two iterations to decide between them. According to my observation, the execution time of ResNet was half VGG16's execution time. As for the accuracy, ResNet was more accurate than VGG16 (although the test loss of VGG16 was less than that corresponding to ResNet). Moreover, I updated the architecture of the ResNet by modifying the number of output features in the last layer (fully-connected layer).


The Final Test Loss = 1.83 and Test Accuracy =  72%
