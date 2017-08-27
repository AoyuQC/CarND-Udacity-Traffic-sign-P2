#**Traffic Sign Recognition** 

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/hist_label.png "Visualization of labels"
[image2]: ./examples/preprocess.png "Preprocess"
[image3]: ./examples/lost_accuracy.png "Training process"
[image4]: ./examples/deer-Copy1.jpg "Traffic Sign 1"
[image5]: ./examples/pedeistrians-Copy1.jpg "Traffic Sign 2"
[image6]: ./examples/signals-Copy1.jpg "Traffic Sign 3"
[image7]: ./examples/turnright-Copy1.jpg "Traffic Sign 4"
[image8]: ./examples/yield-Copy1.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209 
* The size of the validation set is 12630 
* The size of test set is 1024 
* The shape of a traffic sign image is 32x32 
* The number of unique classes/labels in the data set is 43 

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the label set. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided not to convert the images to grayscale because I think the color information is very important for the classifier to learn the features. I also normalize the data. Because it is good to train the model.
Here is an example of a traffic sign image before and after normalization.

![alt text][image2]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is basically the color version of LeNet model. It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16|
| Flatten | inputs 5x5x16, outputs 400|
| Fully connected		| Inputs 400, outputs 120|
| RELU					|												|
| Fully connected		| Inputs 120, outputs 84|
| RELU					|												|
| Fully connected		| Inputs 84, outputs 43|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer from tensorflow. Becuase this works very well when I did LeNet experiments. I chose the learning rate of 0.001 because this is considered as good start point for training. I set the batch size to be 128 and the number of epochs to be 100. I also calculate the cost and accuracy curve during training. This helps me make sure everything goes well. I tried gray input image at first. However, the cost is very high and cannot be reduced very quickly.

![alt text][image3]


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
the cost is 0.000
* validation set accuracy of 0.991 
* test set accuracy of 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
I chose the architecture of LeNet
* Why did you believe it would be relevant to the traffic sign application?
The size for training is very close. The complexity of hand-writing number and traffic sign is not very different
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The cost of final model reaches 0 after 34 epochs. The validation accuracy goes up very quickly. The final test accuracy of 0.94 shows this model has the potential to predict unknown signs.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image should not be difficult to classify because the animal image is very clear. The second image should also not be difficult to classify etheir because it is very clear. The third image may be difficult to classify because it mixed the color signals and triangle size. The fourth image should not be difficult to classify because it looks pretty clear. The final one should be the easiest one because it has the charater of yield in the image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Deer image| No entry| 
| Pedistration| speed limits|
| Signals| General Caution|
| Turn right | Stop					 				|
| Yield| Yield|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This looks really bad. It is hard to understand why it predicts the Deer image as No entry, Pedistration as speed limits, Turn right as stop. The reason that it predicts the Signal image as General Caution may be the triangle shape. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the 1st image, the model is pretty sure that this is a No entry sign (probability of 0.99) but the image does not contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99 | No entry   									| 
| 0.000171 | Slippery road |
| 0.000118892| Right-of-way|
| 4.10706e-07| Speed limit (20km/h)|
| 7.83209e-09| Bicycle crossing|

For the 2nd image, the model is pretty sure that this is a speed limit(30km/h) (probability of 1.0) but the image does not contain a speed limit(30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0| Speed limit(30km/h)| 
| 1.98325e-08| Speed limit(70km/h)|
| 5.75144e-17| General caution|
| 5.26332e-18| Road narrows on the right| 
| 2.80754e-21| Speed limit(20km/h)|

For the 3rd image, the model is pretty sure that this is a General caution(probability of 1.0) but the image does not contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0| General caution| 
| 7.52459e-38|Traffic signals| 
| 0|Speed limit(20km/h)|
| 0|Speed limit(30km/h)|
| 0|Speed limit(50km/h)|

For the 4th image, the model is pretty sure that this is a stop sign(probability of 1.0) but the image does not  contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0| stop| 
| 7.73587e-14|Speed limit(30km/h)| 
| 3.07058e-19|Speed limit(20km/h)|
| 7.44814e-25|Speed limit(80km/h)|
| 2.72469e-25|No entry|

For the 5th image, the model is pretty sure that this is a yiled sign(probability of 1.0) and it is true. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0| Yiled| 
| 1.61486e-41|End of no passing by vehicles over 3.5 meters tons|
| 7.73587e-14|Speed limit(30km/h)| 
| 3.07058e-19|Speed limit(20km/h)|
| 7.44814e-25|Speed limit(80km/h)|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


