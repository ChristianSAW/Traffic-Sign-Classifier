# **Traffic Sign Recognition** 

## Writeup


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

[image1]: ./notebook_images/Visualization_1.jpg "Visualization of Sample Training Image"
[image2]: ./notebook_images/Visualization_2.jpg "Visualization of Sample Training Image from Each Class (43 Total)"
[image3]: ./notebook_images/Training_Hist.jpg "Training Data Set Histogram of Class/Label Distribution"
[image4]: ./notebook_images/Validation_Hist_1.jpg "Validation Data Set Histogram of Class/Label Distribution"
[image5]: ./notebook_images/Testing_Hist.jpg "Test Data Set Histogram of Class/Label Distribution"
[image6]: ./notebook_images/Preprocess_Visualization.jpg "Preprocessing of Images"
[image7]: ./notebook_images/Brightness_Augmentation_Visualization.jpg "Brightness Augmentation of Image (And Preprocessing Applied After)"
[image8]: ./notebook_images/Transformations_Visualization.jpg "Individual And Compound Transformations of Image"
[image9]: ./notebook_images/new_image_Visualization_1.jpg "New Images To Be Tested (No Labels)"
[image10]: ./notebook_images/new_image_Visualization_2.jpg "New Images To Be Tested (With Labels)"
[image11]: ./notebook_images/new_image_predictions.jpg "Predicted Labels of New Images"
[image12]: ./notebook_images/new_image_top_5.jpg "Top 5 Predictions for Each New Image"

---
### Writeup / README

#### 1. Project Code:

Here is a link to my [project code](https://github.com/ChristianSAW/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below can be seen an example of each class/sign type. I manually confirmed that the class ID (#) of each image corresponded to the Sign Name as provided in 'signnames.csv". Also, for each sign type (43 total), I randomly picked one from each class to be displayed. So, if you'd like, you could just continuously generate a visualization of 43 new (probably) signs each time. I would not recommend it though. 

![alt text][image2]

I created a histogram for each of the 3 data sets: training, validation, and testing to better understand how the distribution of sign types was throughout the data sets. Below these hisograms are shown. As you can see, the relative distribution is uneven meaning there are significantly more of certain sign types than others. This could simply be a reflection of how common certain signs are, and if the model is uncertain about a sign, keeping the uneven distribution will mean the model should lean towards the more common sign in the data set. Additionally, the relative uneveness is consistant between the 3 data sets, and it even appears that the relative frequency of sign types is consistant between the data sets. This is good. This decreases the liklihood of the the validation or testing results to be skewed from the different relative number of signs being tested. 
<br> <br>
*TLDR: Relative sign frequency is simmilar between sets, and I wont change this uneven distribution of sign types within the datasets* 

![alt text][image3]

![alt text][image4]

![alt text][image5]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Before I did anything, I did some research to get some inspiration on strategies to employ here. In particular, I found the following three papers/articles very helpful, and I used various approaches from each of them. I tried to give credit where I specifically used a technique in my code.  
<br>
<br>
Articles:
1. "A Committee of Neural Networks for Trafﬁc Sign Classiﬁcation" (http://people.idsia.ch/~juergen/ijcnn2011.pdf) <br>
2. "(98.8% solution) German sign classification using deep learning neural networks" (https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad) <br>
3. "Improved performance of deep learning neural network models for Traffic sign classification using brightness augmentation (99.1% solution)" (https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc) <br>

**I broke down data processing to 2 categories: preprocessing (apply to all images), additional data processing (apply to training images)

#### Preprocessing 

The goal here is to change alter the input image before it enters the model pipeline so that the network can better classify the image. I applied 2 techniques here: pixle normalization, and histogram equalization. Preprocessing is applied to any an all images which go through the nerual net (training or otherwise).

**Pixle Normalization:**
This is where I normalize the pixle images about zero with some range (-0.5 to 0.5 in this case). With this we keep numerical stability: inputs have zero mean and equal variance. I do this so that my optimization problem is well conditioned and does not need to do a lot of searching to find a good solution. 

**Histogram Equalization:**
I apply histogram equalization so the effect of brighness is removed. From reading the  Ciresan et al. paper, I learned of using CLAHE as a more effective alternative. CLAHE differs from traditional histogram equalization by applying adaptive histogram equalization where an image is divided into blocks, and each block is then histogram equalized. Contrast limiting is also applied to avoid amplification of noise in those blocks as a result of the histogram equalization. The essential advantage here is that with CLAHE, you can apply histogram equalization on more confined regions of the image than with regular histogram equalization. In the below images, it's hard to notice the difference between regular histogram equalization and CLAHE, but through testing (and as was found in the Ciresan et al. paper), my training accuracies were consistantly higher with CLAHE.  

Below is Histogram Equalization and CLAHE applied to an image.

![alt text][image6]

Note that no conversion techniques such as converting to greyscale or HSV were used. This is because I planned on implementing a 1x1x3 convolutional layer at the start of the neural net and let the neural net decide what channels were most useful. This was applied in models 2 and 3.

#### Additional Data Processing 

Additional data processing is applied to training images before preprocessing is applied. Essentially, the goal is to modify the image in ways so that during training these images can be recognized as an unmodified image would. With this we can remove the impact of factors like lighting, image rotation angle, image centering, image scaling, etc. Specifically, I examined augmenting images with brightness, rotation, translation, and shear transforms. 

**Brightness Augmentation:**
While the impact of brightness should be mitigated by the histogram equalization, we should still train with images which have lighing issues. Adding brighness augmentation also ensures the neural network does not rely on brighness information even if it isn't midigated. 

Below can be seen the impact of brigness augmentation on an image. For this image, however, there was already of the issue of bad lighting. In the ladder two images Histogram Equalization and CLAHE is applied to the brightness augmented image. It is difficult to see the difference between the augmented image and the Histogram Equalized and CLAHE images. In fact, it seems as if the brighness augmentation in this case improves the lighting such that Histogram Equalization and CLAHE have to do less work. The point here is to see how the brighness augmentation can alter the image. To see differences between Histogram Equalization and CLAHE, the previous images are better to look at.

![alt text][image7]

**Transformations**

I applied 3 types of transformations, rotation, translation, and shearing to represent reflect more realistic images that such a sign classifier might see (and conversly train the neral network to recognize a transformed sign the same way a regular sign). Unlike with brightness, there isn't really any good preprocessing technique we can do to remove this effect of images, so the best we can do is train the network to ignore the effects. Below, you can see the effects of the transformations on a sign. 

![alt text][image8]


I decided against adding additional data, as I beleived I could essentially get the same amount of training by increasing the number of EPOCH, and because I was randomly applying these augmentations and transformations to each batch, in theory each example trained would be different between the EPOCHS. To elaborate, at each EPOCH, I applied data processing to the training data set. Within this processing, for each image, the augmentation, rotation, translation, or shearing was applied at a random factor (within a range); this is how I randomized each image, and made each EPOCH "unique."
<br>
<br>

For each model I tried, I tested different combinations of data processing to determine what worked best. I will share my findings in a later section. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried three different models. The first was almost identical to the LeNet-5 Archetecture. The second built upon this by adding complexity (more layers), dropout, L2 reduction. The third model I tried was an even "larger" version of model 2, meant to accomodate more data processing during training. 

## Initial Attempt Archetecture (Based off LeNet-5)

### Input
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since our images are in color, C is 3 in this case.

### Archetecture 
**Layer 1: Convolutional.** Output shape: [28,28,6]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [14,14,6]

**Layer 2: Convolutonal.** Output shape: [10,10,16]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [5,5,16]

**Flatten** [5,5,16] --> [400]
Flatten converts output from 3D to 1D to transition from filter layers to fully connected layers

**Layer 3: Fully Connected.** Output shape: [120]
<br>
**Activation.** ReLu

**Layer 4: Fully Connected.** Output shape: [84]
<br>
**Activation.** ReLu

**Layer 5: Fully Connected (Logits).** Output shape: [43]

### Output 
Return the result of final fully connected layer (logits)

## Second Attempt Archetecture (model 2) -- **BEST RESULTS FROM MODEL 2**

### Input
Model 2 architecture accepts a 32x32xC image as input, where C is the number of color channels. Since our images are in color, C is 3 in this case.  
<br>
*Note 1: **Below output shapes are what was used for the most sucessful version of model 2 thus far.** 
<br>
*Note 2: Convolutional layers use VALID padding.*

### Archetecture 
**Layer 0: Convolutional 1x1x3.** Output shape: [32,32,3]
<br>
**Activation.** ReLu

**Layer 1: Convolutional.** Output shape: [28,28,64]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [14,14,64]
<br>
**Dropout.** Rate = 0.5, Output shape: [14,14,64]

**Layer 2: Convolutonal.** Output shape: [10,10,128]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [5,5,128]
<br>
**Dropout.** Rate = 0.5, Output shape: [5,5,128]

**Layer 3: Convolutonal.** Output shape: [5,5,256]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [3,3,256]
<br>
**Dropout.** Rate = 0.5, Output shape: [3,3,256]

**Flatten** [3,3,256] --> [2304]
Flatten converts output from 3D to 1D to transition from filter layers to fully connected layers

**Layer 4: Fully Connected.** Output shape: [1024]
<br>
**Activation.** ReLu
<br>
**Dropout.** Rate = 0.5, Output shape: [1024]

**Layer 5: Fully Connected.** Output shape: [1024]
<br>
**Activation.** ReLu
<br>
**Dropout.** Rate = 0.5, Output shape: [1024]

**Layer 6: Fully Connected (Logits).** Output shape: [43]

### Output 
Return the result of final fully connected layer (logits)
 
## Third Attempt Archetecture (model 3)
More complecated version of version 2 to handel more data processing. 
<br>
*Note 1: **Below output shapes are what was used for the most recent use of model 3.** 
<br>
*Note 2: Convolutional layers use SAME padding.

### Input
Model 3 architecture accepts a 32x32xC image as input, where C is the number of color channels. Since our images are in color, C is 3 in this case.

### Archetecture 
**Layer 0: Convolutional 1x1x3.** Output shape: [32,32,3]
<br>
**Activation.** ReLu

**Layer 1: Convolutional.** Output shape: [32,32,64]
<br>
**Activation.** ReLu

**Layer 2: Convolutional.** Output shape: [32,32,64]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [16,16,64]
<br>
**Dropout.** Rate = 0.5, Output shape: [16,16,64]

**Layer 3: Convolutonal.** Output shape: [16,16,128]
<br>
**Activation.** ReLu

**Layer 4: Convolutonal.** Output shape: [16,16,128]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [8,8,128]
<br>
**Dropout.** Rate = 0.5, Output shape: [8,8,128]

**Layer 5: Convolutonal.** Output shape: [8,8,256]
<br>
**Activation.** ReLu

**Layer 6: Convolutonal.** Output shape: [8,8,256]
<br>
**Activation.** ReLu
<br>
**Pooling.** Output shape: [4,4,256]
<br>
**Dropout.** Rate = 0.5, Output shape: [4,4,256]

**Flatten** [4,4,256] --> [262144]
Flatten converts output from 3D to 1D to transition from filter layers to fully connected layers

**Layer 7: Fully Connected.** Output shape: [1024]
<br>
**Activation.** ReLu
<br>
**Dropout.** Rate = 0.5, Output shape: [1024]

**Layer 8: Fully Connected.** Output shape: [1024]
<br>
**Activation.** ReLu
<br>
**Dropout.** Rate = 0.5, Output shape: [1024]

**Layer 9: Fully Connected.** Output shape: [256]
<br>
**Activation.** ReLu
<br>
**Dropout.** Rate = 0.5, Output shape: [256]

**Layer 10: Fully Connected (Logits).** Output shape: [43]

### Output 
Return the result of final fully connected layer (logits)


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used an AdamOptimizer (Better SGD) and the following hyperparameters:

* EPOCH: 50
* batch size: 128
* learning rate: 0.001
* dropout keep probability: 0.5
* L2 regularization beta factor: 0.0005
* learning rate exponential decay factor: 0.96
* accuracy/loss decline percent difference threshold: 0.005

Other factors specific to data processing were also used such as the angle rotation range, translation range, shear range, as well as a decrease factor (since the amount that images were transformed: translation + rotation + shear was decreased at each EPOCH).

Note that all of the hyper parameters were tuned during testing of the model. I will talk about a few. 

* EPOCH: Originally (for Le-Net model and model 2) I used a lower EPOCH of 15, but I expanded it to 30 and eventually 50 as I determined the accuracy had not converged for these models with a lower EPOCH 
* learning rate: For the first model, I tried a few different rates, but settled with 0.001 as a safe number. After I had settled with model 2, I tried using exponential decay with the rate to boost my accuracy, and found that to improve my accuracy. 
* accuracy/loss ... threshold: I initially used a larger number (0.05), but decreased to 0.005 through testing. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.2%
* test set accuracy of 98.4%

#### Approach:


I used an iterative approach to develop my final model. I started with thew well known LeNet-5 archetecture to get a baseline for how well what I would consider a simple model would perform. I was able to get accuracies of 95.3% alone by just tuning this model (hyperparameters as well as the characteristics of each layer). I then made my model larger as I found necessary going through 2 more model iterations. 

#### LeNet-5

**Best Configuration**
* Best Validation Accuracy: 95.3% 
<br>
* Best layer configurations: strides, kernel sizes, etc. can be seen either in the code or the archetecture overview in section 2
<br>
* CLAHE Kernel: 5x5
* Learning rate: 0.001 (no exponential decay)
* EPOCH: 15
* batch size: 128
* pooling kernel size: 2x2
* pooling strides (SAME): 2
* Preprocessing applied to training, validation, test data sets

In the back of my mind, I knew I wanted to have a model that could accomidate as much data processing as possible (brighenss augmentation, transformations: rotation, translation, shear). After applying just the brighness augmentation, validation accuracy decreased (best validation accuracy: 94.5%). And, as I suspected, adding the transformations on top of the brighness augmentation lowered the accuracy even further (best validation accuracy: 81.2%). Simply put, the model was not 'big' enough to accomidate the amount of data processing that I desired. Additionally, the model began to overfitt with this additional data processing, so I knew I had to add dropout and L2 regularization in the next model to midigate this. 

**Why start with LeNet-5?**
LeNet-5 is a historically sucessful implemntation of a CNN for this problem. The convolutional layers work well with this problem because it allows the nerual net to see the feature beyond a pixle and the image as a whole. With the convolution, the neural net can 'see' the relation of one pixle with its neighbors. Additionally, convolutional layers here allow for spatial invariance so each filter can extract a feature throughout the whole image and not just in one location. 

**Parameter Tuning:**
Pooling was not tuned. Only EPOCH, batch size, learning rate, and CLAHE kernel size were tuned. <br>
* EPOCH: Initially I started with 10 EPOCH. I tested for higher EPOCH and setteled with 15 (only tested with preprocessinig applied to training set) as the accuracy consistantly converged by 15 EPOCH. Looking back, I should have used a even higher EPOCH as additional data processing needed more iterations. However, this model did not have any stops for overfitting as later models would.  
* Batch size: I tested batch sizes {50, 100, 128, 150, 200, 256}. Quickly I settled on 128. Simply put, 128 yielded the best validation accuracies. 
* learning rate: I started with a higher learning rate, 0.1 and lowered it until I saw the lowest loss before loss increased again (e.g. I tried to find the global min). I setteled with a learning rate of 0.01.

#### Model 2

**Best configuration -- THIS WAS ALSO THE OVERALL BEST CONFIGURATION THAT YIELDED THE BEST VALIDATION AND TESTING ACCURACY**
* Best Validation Accuracy: 98.2%
<br>
* See the code for the exact final parameters and hyperparameters
* CLAHE Kernel: 4x4
* Learning Rate: 0.001 (with exponential decay)
* Dropout keep probability: 0.5
* L2 regularization beta factor: 0.0005
* Learning rate exponential decay factor: 0.96
* Accuracy/loss decline percent difference threshold: 0.005
* EPOCH: 50
* batch size: 128
* pooling kernel size: 2x2
* pooling strides (SAME): 2
* Preprocessing applied to training, validation, test data sets
* Brightness augmentation applied to training. 

To accomidate the additional data processing, I increased the model size by adding 2 additional convolutional layers and increasing the the overall outputs of each convolutional and fully connected layer. I added dropout and L2 regularization to account for the overfitting seen in the last model. Initially, I did not use learning rate decay or use such a large EPOCH. I developed model 3 after this inital look at model 2 and then went back to model 2 and developed it further. This is when I increaed the EPOCH and added the decay. 

**First Look at Model 2:**
Here I added the 1x1x3 convolutinal layer at the beginning (as I had intended) so that the nerual net could chose the best channels for the input images rather than deciding that myself with some preprocessing. I tested the following: 
1. EPOCH
2. # of filters for convolutional layers 1-3
3. # of outputs for fully connected layers 4-6
4. What data processing to be applied to the trainiing data set?
5. Transformation parameters: rotation range, translation range, shearing range 
6. Use of CLAHE or not
7. L2 beta rate

**Parameter Tuning:**
I will now go through each of these. <br><br>
1: I increased the EPOCH from 15 to 30 arbitrarily and found that my accuracy was still increasing. As I did later, I should have increased it even further. Later, in my second look at model 2, I tested for the EPOCH where accuracy stagnated. 
2: Starting with the same # of filters from LeNet-5, I incrementally increased them. I found increasing the filter size here to have the first significant impact. Essentially doubling the filter sized changed my validation accuracy at 15 EPOCH from 36% to 87.3%. I increased the filter sizes to {L1: 64, L2: 128, L3: 256}. The increases between these values and the previous ones I used were small, so I believed further increases would have negligible impact. 
3: Again, starting with the low number of outputs from LeNet-5, I found increasing them slighly helped the accuracy of the model. 
4: I found that this model could handle brighness augmentation (best validation accuracies in first attempt), but it could not handle transformations as well. Both achieved final accuracies above 93%, but only using brighness augmentation still yielded better results. 
5: I used initial low transformation ranges {rotation range: 10 deg, translation range: 2 pixles, shear range: 2 pixles}, and tried training with significantly higher values that would be more realistic {rotation range:  90 deg, translation range: 10 pixles, shear range: 5 pixles}, but found using such high ranges adversly affected my validation accuracies (They went from 93.2% to 27.7%)
6: As I did with LeNet-5, I tested using either Historgram Equalization or CLAHE, and found CLAHE to again yield better accuracies. 
7: I don't have much intution with beta values, so I looked at Vivek Yadav's model to see what he used. I used a slighly larger value, and then lowered it to see if the accuracies were better -- they were. In the future, this is definatly a parameter I could tune more. 

**Second Look at Model 2:**
After trying and failing with model 3, I revisited model 2. The first thing I did differently was significantly raise the EPOCH number to see if the accuracy would increase anymore. This turned out to be a good move. Simply doing this gave me a 2% increase in my accuracy. I also introduced decay of the learning rate when the training loss stagnates. I did not tune the decay rate; I used the default rate that the built in tensorflow function (tf.train.exponential_decay()) used. I did, however not use this function so I could establish the condition when the learning rate should be lowered. Essentially, I exponentially lowered the learning rate whenever the % difference in accuracies between the current EPOCH and the prevous EPOCH was less than a threshold. I tuned the model to find the ideal threshold for my model parameters and hyperparameters that I had already established. I fould a threshold of 0.005 to work just right. Once I had the model I liked, I increased the EPOCH to 150 to see if the accuracies could increase even further. It turns out, that by 50 EPOCH, the accuracy had coverged. 

#### Model 3

**Best configuration:**
* Best Validation Accuracy: 52%
<br>
* See the code for the exact final parameters and hyperparameters
* CLAHE Kernel: 4x4
* Learning Rate: 0.001 
* Dropout keep probability: 0.5
* L2 regularization beta factor: 0.0005
* EPOCH: 30
* batch size: 128
* pooling kernel size: 2x2
* pooling strides (SAME): 2
* Preprocessing applied to training, validation, test data sets
* Brightness augmentation applied to training. 

Because model 2 did not seem large enough to accomidate for transformations of the image, I created an even larger model, adding three convolutional layers and an additional fully connected layer. Unfortunately, I was unable to find any parameters or hyperparameters that would make this model usable. I will revisit this model at somepoint, make some modifications, because I do think a more complex version of model 2 is necessary to get transformed images to be able to be trained properly. It is also entirely possible that I simply need more data and training. Unfortunately, model 3 training times were very long (approximatly 30 minutes), so I did not do as much testing here before I moved back to model 2. 

**Paramer Tuning**

The most sucessful run of this was with the configuration seen in the code. For essentially every configuration I tried thereafter I couldn't get the validation accuracy over 8% with a few exceptions. I tuned the following:

1. # of filters for convolutional layers 1-6
2. # of outputs for fully connected layers 7-9
3. Transformation parameters: rotation range, translation range, shearing range 
4. Convolutional padding, SAME or VALID.

Unfortunately, the accuracy results started around 5% (first EPOCH) for almost every test and remained there throughout all 30 EPOCH so it was difficult to see the impact of any parameters I tuned. 

#### Conclusion:
While I could not find an effective way to use transformations of the image, I still got quite good results: Trainiing Accuracy: 100%, Validation Accuracy: 98.2%, Testing Accuracy: 98.4%. I know that to improve these further, I need to find a way to incorporate data transformations in my data processing. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used 12 images (2 of which were not within the class lables).

![alt text][image10] 

Image's 0, 5, and 8 might be difficult to classify because of the glare that is concentrated in the images. Image 4 is sheared quite a bit, and my model did not train with intentionally sheared images, so that image too might be difficult to classify. Images 1 and 9 will be impossible to classify because they are not within the list of possible lables. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction (Note the image numbers are not the same as the previous image):

![alt text][image11] 

Of the 10 signs that the model should be able to classify, the neural net correctly classified 8 of them. It incorrectly classified 2 signs: <br>
* Sign: 'Speed limit (60km/h),' predicted: 'Speed limit (80km/h)'
* Sign: 'Ahead only,' predicted: 'Go straight or left' 

80% accuracy for new signs is not terrible. Even the signs that it missclassified were relatively close to the right sign. The 'Speed limit (60km/h)' sign that was misclassified was still classified as a speed limit sign. It was also classified as a 80km/h sign and a 6 looks the most simmilar to an 8 compared to the other 4 numbers for double digit speed signs. The 'Ahead only sign' that was misclassified was also classified to a sign that is relatively simmilar and has many of the same characteristics. 

Even the two signs that were not able to be classified correctly were classified as signs that are simmilar. The 'Speed limit (40km/h)' sign was classified as a speed limit sign. Its difficult to say if 80km/h is the closest looking speed sign. The 'No u-turn' sign was not classified as well as the ladder. A better classification would probably have been the 'Turn left ahead' sign, but there is not sign that has all the characteristics of this sign. Still, the curved arrow was what probably led the model to predict the day it did. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below are the top 5 softmax probablilities for each sign. Most of the probabilities are highly certain. I will talk about the ones that were not and were incorrect. 

![alt text][image12] 

**Speed limit (40km/h)**
As expected, there is uncertainty with this sign since there is not correct label. The highest certainty (~55%) was for the 80km/h sign followed by 60km/h (~29%). Still all top five probabilities are speed signs, so the neural net is at least highly certain it is a speed sign 

**Ahead only**
The neural net is ~100% certain the sign is 'Go straight or right.' While the second highest option was the correct label, there is little to no certainty from the nerual net that the sign is 'Ahead only.' 

**Speed limit (20km/h)**
The neural net is over 80% certain that the sign is a 'Speed limit (20km/h)' sign, which it is. The other option it was considering was 'Speed limig (30km/h)' at around 17% certainty. 

**Speed limit (60km/h)**
The nerual net is over 90% certain that the sign is a 'Speed limig (80km/h)' sign, which is incorrect. The second closest option was the correct label, but only with a confidence at around 4%. 

**No u-turn**
This is probably the most interesting sign to be classified that couldn't be classified correctly because it was not distinctly close to one sign. It had features from several different. This classification also had the lowest confidence. The chosen prediction was with a confidence of less than 50% (~46%). The other top 5 labels had between 5 and 10% confidence. The nerual net saw features of this sign similar to the 'Children crossing' sign (~10%) -- possibly because of the colors; Speed limit (60km/h)' sign (~7%) -- possibly because of the shape and curvature of the arrow being similar to the curvature of a number; End of speed limit (80km/h) sign (~6%) -- possibly because of the red slash through the sign; and 'Ahead only' sign (~6%) -- possibly because of the arrow. 

