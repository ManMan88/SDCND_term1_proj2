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
* Extra: visualize some layers of the neural netwotk


[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/original.png "Original"
[image3]: ./images/grayscale.png "Grayscaled"
[image4]: ./images/processed.png "Processed"
[image5]: ./new_images/new_images.png "New Images"
[image10]: ./images/noentry_conv1.png "No Entry conv1"
[image11]: ./images/noentry_conv2.png "No Entry conv2"
[image12]: ./images/work_conv1.png "Work conv1"
[image13]: ./images/work_conv2.png "Work conv2"

## Rubric Points
##### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
---
#### 1. Data Set Exploration
##### Summary
I used python to calculate summary statistics of the traffic signs data set:
* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3) -> 32x32 pixels with 3 color channels
* The number of unique classes/labels in the data set is 43

##### Visualization

Here is an exploratory visualization of the data set. I used numpy and matplotlib to calculate the statistics of the data and visualize it.
The following image is a bar chart showing the fraction of images of each traffic-sign class out of the total images it the relevant set. For example: traffic-sign number 0: the training set contains ~0.005 = 0.5% images of this traffic-sign.
The green, red and blue bars represent the training, validation and testing sets correspondingly.

![Visualization][image1]

It is clear that the fractions of each class is simillar in all sets. However, there are classes that apears a lot more compared to others. This will probably affect the trained neural network to identify the larger classes with greater success. 

#### 2. Design and Test a Model Architecture

##### Image processing
In this secion I'll describe the image processing method I used. I'll show an example of 5 processed images. 
Here are the he original images:
![Original][image2]

**Grayscale and Normalization**
First, I converted all images to grayscale. After experimenting with both archituctures for 3 color channels and grayscale channel, I concluded that there is no benefit for using the 3 color channels as input. The predictions of the validation set weren't improved for the 3 color channels as input. Furthermore, as written in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the grayscale images teilded better results. In order to convert to grayscale, I converted the images to YUV fromat and used the Y value.
Second, I normalized the images to the values between -1 and 1. This was done in order to prevent scaling factors between different data samples. It also helps preventing numerical errors and making the traning process faster.
Here are the 5 grayscaled and normalized images:
![Grayscale][image3]

**Generating Additional Data (was not implemented)**
This [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggests to add more training data by using the current data and applying some processing such as: translating, rotating, scaling and adding noise to the grayscaled images. This is done so the neural network will be robust and won't be affected by these characteristics.
I tried using this approach but it yeilded no better results. Insted I processed the training data similarly to processing of the added data.

**Translating, Rotating, Scaling and Adding Noise**
The training images were translated, rotated, scaled and added noise to. The scale of translation, rotation, scale and added noise was randomized with a truncated gaussian distribution. The max values were chosen similarly to the values used in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
* The translation max is 3px (width and height each) with std of 1
* The rotaion max is 15deg with std of 5
* The scaling max is 1+-0.1 with std of 0.04
* The noise wasn't truncated, with std of 0.01

Here are the processed images:
![Processed][image4]

##### Model Architecture
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x16 	|
| Pooling 2x2        	| 2x2 stride, valid padding, outputs 14x14x16 	|
| RELU	        		| outputs 14x14x16								|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					| outputs 10x10x32								|
| Perception    	    | Combination of: 1x1 conv, pooling with 1x1 conv, 1x1 conv followed by 6x6 conv, 1x1 conv followed by 2x2 conv. outputs 5x5x64|
| Flattening    		| width 1600        							|
| Fully connected		| width 533       						    	|
| Fully connected with dropout = 0.5	|	width 266					|
| Fully connected with dropout = 0.5	|   width 43					|
| SoftMax	                            |   width 43					|
 
I chose this model (also using the perception model) after drawing inspiration from [google's papaer](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf).
I tried several archituctures with different depth of layers, and this one performed the best for this task.

##### Training Process
To train the model, I used the one hot encoding method, and then calculated the cross entropy. Then the neural network's weights and biases were optimized using the Adam algorithm optimizer.
The parameters for the training process were chosen after trial and error as follows:
* batch size: 512
* Number of epochs: 10
* Learning rate: 0.001
* Dropout (where mentiond): 0.5

##### Solution Approach
First, I looked into this paper [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and used its stated guidlines. However, the neural network architucture was chosen as a combination af LaNet with a perception layer (the perception laer was inspired from this paper: [google's papaer](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)). I tried several other archituctures until setteling on the current one, examaning them by the training and validation accuracy. I started with LaNet; then a similar architucture to the first paper; then I tried the current achitucture; then I tried using the RGB data as input (3 feature maps insted of one); and lastly I went back to the current architucture. 
At first, the architucture training set loss was not good enough, so I added more depth (more feature maps). Now the neural network was slightly over fitting the training data, so I added the dropout step which solved the problem.
After choosing the architucture, I further tuned the parameters of the training process such that good results were obtained without over fitting the training data. After achieving satisfying results in the training process, the test data was evaluated.

My final model results were:
* training set loss of 0.017
* validation set accuracy of 0.967 
* test set accuracy of 0.949

#### 3. Test a Model on New Images

##### Chosen 5 images
Here are five German traffic signs that I found on the web:
![New Images][image5]

Each image has some difficulties to be classified as follows:
* stop sign: the angle of the sign
* roundabout: noisy image
* road work: the angle of the sign
* no entry: the text in the middle of the sign
* double curve: not an original german traffic sign

##### The Model's Predictions
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop sign      		| Stop sign   									| 
| Round about     		| Priority road 								|
| Road work				| Road work										|
| No entry	      		| No entry					    				|
| Double curve			| Double curve      							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares similarly to the accuracy on the test set, but it cannot be compared since this set is very small and is not representative.

##### Model's Certainty 
For the first image, the model is not quite sure that this is a stop sign (probability of 0.616), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .616         			| Stop sign   									| 
| .384     				| Speed limit (30km/h) 										|
| .0					| Turn left ahead											|
| .0	      			| Turn right ahead					 				|
| .0				    | Roundabout      							|

For the second image, the model is not quite sure that this is a priority road (probability of 0.597), but the image contains a roundabout sign. The second probability (0.388) was of a roundabout. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .597         			| Priority Road   									| 
| .388     				| Roundabout 										|
| .011					| Speed limit (100km/h)										|
| .002	      			| Speed limit (50km/h)					 				|
| .001				    | Yeild     							|

For the third image, the model is very sure that this is a road work sign (probability of 0.989), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .989        			| Road work   									| 
| .007    				| Bicycles crossing 										|
| .004					| Bumpy road											|
| .0	      			| Road nurrows on the right					 				|
| .0				    | Wild animals crossing      							|

For the fourth image, the model is very sure that this is a no entry sign (probability of 1.0), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.        			| No entry   									| 
| .0    				| Stop sign 										|
| .0					| Turn left ahead											|
| .0	      			| Turn right ahead					 				|
| .0				    | Roundabout      							|

For the fifth image, the model is very sure that this is a double curve sign (probability of 0.913), and the image does contain a double curve sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.913        			| double curve   									| 
| .083    				| Road nurrows on the right 										|
| .003					| Dangerous curve to the left											|
| .0	      			| Wild animals crossing					 				|
| .0				    | General caution      							|
#### Visualizing the Neural Network 
##### The first Convolution Layer
**No Entry Sign**
The results for the no entry sign from the 5 images:
![NO Entry conv1][image10]
**Road Work Sign**
The results for the road work sign from the 5 images:
![Road Work conv1][image11]
It seems that the first layer looks for outlines. It recognizes staright and curved lines in the image. One can clearly see the circle and middle line for the no entry sign, and the triangle lines of the road work sign.

##### The second Convolution Layer
**No Entry Sign**
The results for the no entry sign from the 5 images:
![NO Entry conv2][image11]
**Road Work Sign**
The results for the road work sign from the 5 images:
![Road Work conv2][image12]

It is less clear what does the second convolution layer recognizes, but it seems to be looking for some features inside the sign. For the no entry sing, there are some clear contrast images of the midle line and surroundings. For the road work sign, it seems to emphasize the working man in the middle of the sign in some feature maps.  


