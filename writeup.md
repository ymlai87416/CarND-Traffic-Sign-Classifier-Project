# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train_ds_dist]: ./write_up/data_dist_train_set.png "Train data set distribution"
[valid_ds_dist]: ./write_up/data_dist_valid_set.png "Valid data set distribution"
[test_ds_dist]: ./write_up/data_dist_test_set.png "Test data set distribution"
[image_augment]: ./write_up/image_generator.png "original image and an augmented image"
[before_grayscaling]: ./write_up/before_grayscaling.png "Before grayscaling"
[after_grayscaling]: ./write_up/after_grayscaling.png "After grayscaling"
[lenet_dropout_0.5]: ./write_up/lenet_dropout_0_5.png "LeNet for dropout = 0.5"
[lenet_dropout_0.3]: ./write_up/lenet_dropout_0_3.png "LeNet for dropout = 0.3"
[lenet_dropout_0]: ./write_up/lenet_dropout_0.png "LeNet for dropout = 0"
[lenet_dropout]: ./write_up/lenet_dropout.png "LeNet dropout"
[mscnn_dropout_0.5]: ./write_up/mscnn_dropout_0_5.png "MsCNN for dropout = 0.5"
[mscnn_dropout_0.4]: ./write_up/mscnn_dropout_0_4.png "MsCNN for dropout = 0.4"
[mscnn_dropout_0]: ./write_up/mscnn_dropout_0.png "MsCNN for dropout = 0"
[mscnn_dropout]: ./write_up/mscnn_dropout.png "MsCNN dropout"
[lenet_confusion_matrix]: ./write_up/lenet_confusion_matrix.png "LeNet confusion matrix"
[mscnn_confusion_matrix]: ./write_up/mscnn_confusion_matrix.png "MsCNN confusion matrix"
[image1]: ./test_input/Bicycles_crossing.png "Bicyclecrossing"
[image2]: ./test_input/Go_straight_or_right.png "Gostraight or right"
[image3]: ./test_input/No_passing_for_vehicles_over_3.5_metric_tons.png "No passing for vehicles over 3.5 metric tons"
[image4]: ./test_input/predestrian.png "predestrian"
[image5]: ./test_input/Priority_road.png "Priority road"
[top5_1]: ./write_up/top5_1.png "top5 1" 
[top5_2]: ./write_up/top5_2.png "top5 2" 
[top5_3]: ./write_up/top5_3.png "top5 3" 
[top5_4]: ./write_up/top5_4.png "top5 4" 
[top5_5]: ./write_up/top5_5.png "top5 5" 
[feature_map_layer1]: ./write_up/feature_map_layer1.png "Feature map layer 1" 
[feature_map_layer2]: ./write_up/feature_map_layer2.png "Feature map layer 2" 
[lenet_train_30]: ./write_up/lenet_train.png "LeNet training" 
[mscnn_train_30]: ./write_up/mscnn_train.png "Multi-scale cnn training" 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/ymlai87416/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 X 32 with RGB channel
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number for each traffic sign in the dataset

Across train dataset

![alt text][train_ds_dist]

Across valid dataset

![alt text][valid_ds_dist]

Across test dataset

![alt text][test_ds_dist]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

As a first step, I decided to make the number of images per class equals, because I don't want the neural network to learn about the distribution of class in the train dataset.

Keras provides a good generator "ImageDataGenerator" to generate augmented data from original data. Using augmented data, the neural network is more resilient to affine transformation and also the brightness of an image, which does not affect the meaning of the sign.

The range of rotation is between [-15 degrees to 15 degrees], zooming from [0.9 to 1.1], shifting from both vertical and horizontal axis by 3 pixels, I do not consider horizontal flip and vertical flip as it may change the meaning of the traffic sign.

For each class, I take only the first 4000 images from combination of the original dataset and images generated.

Here is an example of original images and augmented images:

![alt text][image_augment]

As a second step, I normalize in each color channels and apply grayscaling to the image.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][before_grayscaling]

![alt text][after_grayscaling]

As the last step, I convert the image from [0 to 255] to [-1 to 1], because large values will result in overflow when calculating weight updates in the forward or backward propagation and make tuning learning rate a harder works.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

In this project, I have created 2 models. The first one is LeNet and the second one is the Multi-scale CNN.
Doing so help me to draw a baseline and evaluate if the additional effort is worth investing in creating a more complex network.

My final LeNet model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   				    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6.  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x6 	|
| Fully connected		| inputs 400, output 120, dropout rate 0.3      |
| Fully connected		| inputs 120, output 84, dropout rate 0.3     	|
| Fully connected		| inputs 84, output 43                        	|
| Softmax				|           									|
 
 
 My final Multi-scale CNN model consisted of the following layers:

| Layer         		|     Description	        					            | 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   				                | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108          	|
| RELU					|											            	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x108 			            	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x108 	            |
| RELU					|												            |
| Max pooling	      	| 2x2 stride,  outputs 5x5x108 				                |
| Fully connected		| input 14x14x108 + 5x5x108, output 100, dropout rate 0.4   |
| Fully connected		| input 100, output 100, dropout rate 0.4                   |
| Fully connected		| input 100, output 43                                      |
| Softmax				|           									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

##### LeNet model
To train the  model, I first search for a learning rate which suitable, the default learning rate 0.001 is quite good.

For the batch size, I choose 128, because it can reduce the memory footprint and also allow frequent updates to weights compare to the big batch size like 2048, and hence can often reach accuracy > 0.8 at the 1st epoch.

Then I searched for the dropout rate for the network. My choice is from 0.5 to 1.

![alt text][lenet_dropout_0.5]
![alt text][lenet_dropout_0.3]
![alt text][lenet_dropout_0]

Studying the graph below find out that for dropout rate = 0, the model has a great flexibility and resulting in a high training accuracy but having a low validation accuracy. but if the dropout rate is too high, the model starts to cripple and cannot perform well in neither the validation dataset nor the training dataset.

At around 0.3, the training accuracy is more or less the validation accuracy, it should a good for the bias-variance tradeoff.
 
![alt text][lenet_dropout]

In this project, I have chosen 0.3 as my dropout rate for the LeNet model.

For the number of epochs used in training the model, I use 30 epochs. Although the time is longer, I can use a smaller learning rate and a bigger dropout rate to aims for a better model.

##### Multi-scale CNN
To train the  model, I first search for a learning rate which suitable, the learning rate I choose is 0.3 with epsilon=1.

For the batch size, I choose 128, reason same as above.

Then I search for the dropout rate for the network. 

![alt text][mscnn_dropout_0.5]
![alt text][mscnn_dropout_0.4]
![alt text][mscnn_dropout_0]

Study the below image arrive more or less the same conclusion of that of LeNet model, From the graph, I can choose either 0.3 or 0.4 as the dropout rate as the network starts not to overfitting the training set and the validation accuracy improves. 

![alt text][mscnn_dropout]

Considering this is a big network, I have chosen 0.4 as my dropout rate.

For the number of epochs used in training, I use 30 epochs. The same reason above.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

LeNet
* training set accuracy of 98.5%
* validation set accuracy of 97.5%
* test set accuracy of 94.9%

Multi-scale CNN
* training set accuracy of 99.7%
* validation set accuracy of 98.3%
* test set accuracy of 96.3%

Explanations:
* What was the first architecture that was tried and why was it chosen?
    * I have chosen LeNet as my first architecture as it is easier to implement and it is also used in lecture video, a great starting point for this project. But without applying other techniques like data augmentation and dropout, this network can only perform at best 92%, which is not useful. Then I turned to Multi-scale CNN, which does provide over 93% in a vanilla setting.
* What were some problems with the initial architecture?
    * The problem of initial architecture is that it is a simple model. It needs optimization to perform its best, and even optimization cannot beat the-state-of-art performance, which is required for a self-driving car.
    * Despite the problems, it does run fast. It consumes half the runtime to complete the training compare to Multi-scale CNN model.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates overfitting; a low accuracy on both sets indicates underfitting.
    * The LeNet did show it is an underfitting architecture, with data augmentation, accuracy only raises to 92%-93%. which make the success of project a matter of lucks. Compare to human 98.81%, a lot of work has to be done if I continue to choose LeNet as my final answer. For that, I searched for a better model for this project.
    * Using Multi-scale CNN is like a big-gun as the model size is of 32MB, compare to LeNet of 800KB. It offers promising result by feeding it the original dataset, and the accuracy continues to grow as I throw in data augmentation and dropout to make it no overfitting to training data set.
* Which parameters were tuned? How were they adjusted and why?
    * Learning rate is quite hard to tune. Too big and too small make the network does not learn at all.
    * For number of epochs, I plot a graph of training accuracy vs validation accuracy. For 30 epochs, I do not see the model to overfit the training set, and I think that 10-20 minutes is worth for training such a network.
    * For dropout, I spent the time to evaluate different parameter, plot the training accuracy and validation accuracy using different dropout ratio, and find a rate which I think it is a bias-variance tradeoff.
    * For batch size, I used batch size=2048 at the beginning, but the infrequent update of weights compare to batch size=128 make the training slow.

![alt text][lenet_train_30]
![alt text][mscnn_train_30]
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * Convolution layer is the state-of-art technique to deal with computer vision problem, it is rooted from how the human brain works when seeing an image. It learns the features automatically without humans to discover the features and hardcode it to the program, which requires time and effort, it can also enable a technique called transfer learning, which reuses the base layer of convolution network for another image classification problem.
    * Dropout layer help to create a successful model because it forces the neural network to learn multiple independent representations of the same data, which prevent the neurons from co-adapting too much with each other, which make overfitting unlikely to happen.

If a well known architecture was chosen:
* What architecture was chosen?
    * I have chosen the multi-scale CNN as my final model.
* Why did you believe it would be relevant to the traffic sign application?
    * This network is used as a submission to a German traffic sign competition GTSRB. This is the most relevant network I can find for this application. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    * My reimplementation of the model is near to what the model in the paper has claimed  98.97% vs 96.3%. I think this network is working well. One problem is that the error for "Pedestrians" is too high, failure rate reaches 50%, which is not acceptable.

#### Confusion matrix for the proposed models
Both models achieve over 93% accuracy, and correctly identify most of the traffic signs, except one, the "Pedestrians" traffic sign, both having around 50% of accuracy rate. Further study must be conducted to find out why the sign is likely to be wrongly classified. 

##### Lenet
![alt text][lenet_confusion_matrix]

##### Multi-scale CNN
![alt text][mscnn_confusion_matrix]

### Test a Model on New Images

#### 1. Choose 5 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 5 German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction made by Multi-scale CNN model:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycles crossing     | Bicycles crossing   							| 
| Go straight or right 	| Go straight or right 							|
| No passing for vehicles over 3.5 metric tons| No passing for vehicles over 3.5 metric tons|
| Pedestrians	      	| Pedestrians       					 		|
| Priority road     	| Priority road                                 |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares to the accuracy on the test set of 96.3%

#### 3. Describe how certain the model is when predicting on each of the 5 new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is sure that this is a Bicycles crossing sign, and the image does contain a Bicycles crossing sign. The top five softmax probabilities were

![alt text][top5_1] 

For the second image, the model is sure that this is a Go straight or right  sign, and the image does contain a Go straight or right  sign. The top five softmax probabilities were

![alt text][top5_2]

For the third image, the model is sure that this is a "No passing for vehicles over 3.5 metric tons" sign, and the image does contain the sign. The top five softmax probabilities were

![alt text][top5_3] 

For the third image, the model is pretty sure that this is a Pedestrians sign, and the image does contain a Pedestrians sign. The top five softmax probabilities were

![alt text][top5_4] 

For the fifth image, the model is sure that this is a Priority road sign, and the image does contain a Priority road sign. The top five softmax probabilities were

![alt text][top5_5] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The below image is the trained network's feature maps when I input a "Bicycle crossing" sign, in the feature map, we can see that neural network did show full or partial triangles, within the triangle, sometimes it has a bicycle shape in it. For some image, the neural network did use horizontal line behind the triangle to identify the sign, but this should not be the case.

For the second layer, it reduces to dots, which I cannot interpret anything meaningful from it.

![alt text][feature_map_layer1] 
![alt text][feature_map_layer2] 
