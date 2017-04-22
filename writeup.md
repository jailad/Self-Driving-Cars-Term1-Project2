#**Traffic Sign Recognition project submission by Jai Lad** 

[//]: # (Image References)

[image1]: ./NewImages/5.gif "5.gif"
[image2]: ./NewImages/9.gif "9.gif"
[image3]: ./NewImages/10.gif "10.gif"
[image4]: ./NewImages/13.gif "13.gif"
[image5]: ./NewImages/22.gif "22.gif"
[image5]: ./NewImages/25.gif "25.gif"
[image6]: ./NewImages/26.gif "26.gif"
[image7]: ./NewImages/27.gif "27.gif"
[image8]: ./NewImages/unknown.gif "unknown.gif"
[image9]: ./NewImages/unknown2.gif "unknown2.gif"

[image10]: ./Traffic_Sign_Classifier_submit/output_47_1.png "output_47_1.png - Training data - Distribution of number of training samples for classes"
[image11]: ./Traffic_Sign_Classifier_submit/output_36_8.png "output_36_8.png - Training data - Wild animals crossing"
[image12]: ./Traffic_Sign_Classifier_submit/output_36_9.png "output_36_9.png - Training data - Keep right"

[image13]: ./Traffic_Sign_Classifier_submit/output_106_2.png "output_106_2 - Misclassified image from the new image set"
[image14]: ./Traffic_Sign_Classifier_submit/output_106_3.png "output_106_3 - Misclassified image from the new image set"
[image15]: ./Traffic_Sign_Classifier_submit/output_106_4.png "output_106_4 - Misclassified image from the new image set"
[image16]: ./Traffic_Sign_Classifier_submit/output_106_5.png "output_106_5 - Misclassified image from the new image set"

---

# Table of contents

1. [Objective](#objective)
2. [Key Files](#keyfiles)
2. [Load the data set](#loaddata)
3. [Explore, summarize and visualize the data set](#exploredata)
    1. [Explore data](#exploredata1)
    2. [Summarize data](#exploredata2)
    3. [Visualize data](#exploredata3)
    4. [Deciding under representation and over representation in training data](#exploredata4)
    5. [Training data set - under represented classes](#exploredata5)
    6. [Data augmentation](#exploredata6)
4. [Preprocess the data set](#preprocessdata)
5. [Design, train and test a model architecture](#designmodel)
    1. [Design a model architecture](#designmodel1)
    2. [Model layers](#designmodel4)
    3. [Train the model](#designmodel2)
    4. [Test the model](#designmodel3)
    5. [A note on using Amazon GPUs for training](#designmodel5)
    6. [Evaluating prediction failures in training data set](#designmodel6)
    7. [Evaluating prediction failures in validation data set](#designmodel7)
    8. [Evaluating prediction failures in testing data set](#designmodel8)
    9. [Final Results](#designmodel9)
6. [Performance against new images](#newimageperformance)
    1. [Acquire new images](#newimageperformance1)
    2. [Test the performance against new images](#newimageperformance2)
    3. [Misclassified images](#newimageperformance3)
    4. [Output the top 5 softmax probabilities for these images](#newimageperformance4)
7. [Retrospective](#retro)
    1. [What went well](#retro1)
    2. [Identifying areas for improvement and learning](#retro2)
8. [Additional exploration](#additionalexploration)
9. [Additional Resources](#additionalreading)
    1. [CNN Architectures](#additionalreading1)
    2. [Tensorflow best practices](#additionalreading2)
10. [(Optional) Visualization of activation](#activationvisualization)

---

### Objective <a name="objective"></a>

The goals / steps of this project are the following:

*  Load a data set of German traffic signs.

*  Explore, summarize and visualize the data set.

*  Preprocess the data as needed.

*  Design, train and test a model architecture, to achieve a validation accuracy of at least 93%.

*  Use the model to make predictions on new images.

*  Analyze the softmax probabilities of the new images.

*  Summarize the results with a written report.

---

### Key Files <a name="keyfiles"></a>

The key files for this project are:

* [writeup.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/writeup.md) - Detailed project write-up.

* [Traffic_Sign_Classifier_submit/Traffic_Sign_Classifier.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_submit/Traffic_Sign_Classifier.md) - Fully executed Python notebook - Markdown format.

* [Traffic_Sign_Classifier_submit.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_submit.ipynb) - Fully executed Python notebook - iPynb format.

* [Traffic_Sign_Classifier_submit.html](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_submit.html) - Fully executed Python notebook - HTML format.

* [Traffic_Sign_Classifier_dev.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_dev.ipynb) - The actual developmental Python notebook from which the above submissions were exported.


---

### - Load the data set <a name="loaddata"></a>

* Downloaded the data set from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).

* Placed those within the folder 'traffic-signs-data'.

* Loaded these pickle file(s) from the notebook.

---

### - Explore, summarize and visualize the data set <a name="exploredata"></a>

#### - Explore the data set <a name="exploredata1"></a>

* Explored the sizes of the training, validation and test sets.

* Explored random images from training, validation and test sets.

* Explored the shape of a single image from the training set - 32 x 32 x 3.

* The above is useful to decide on the architecture of the input layer of a CNN.

* Randomly checked image samples from training set, against their provided labels to confirm that they are labelled correctly. For example, some images from the training set are provided below :

![alt text][image11]

![alt text][image12]

* For the various data sets, confirmed that the number of samples was equal to the number of labels. 

#### - Summarize the data set <a name="exploredata2"></a>
<BR>


| Statistic         	|     Number	        						| 
|:---------------------:|:---------------------------------------------:| 
| # Training examples   | 34799   										| 
| # Training examples 	| 4410	 										|
| # Training examples	| 12630											|
| Image Data Shape		| 32x32x3										|
| # Unique Labels		| 43											|

<BR>


#### - Visualize the data set <a name="exploredata3"></a>

* By intuition, the neural network will train better for the classes for which we have more sample data, versus those for which we have less sample data.

* Distribution of number of samples, by class ( sign category ) for training data

![alt text][image10]

* Statistics below for distribution of number of samples, by class ID
<BR>

| Statistic         	|     Number	        						| 
|:---------------------:|:---------------------------------------------:| 
| Mean         			| 809   										| 
| Std Dev  				| 619.42	 									|
| Max					| 2010											|

<BR>

#### - Separating into under represented and over represented classes <a name="exploredata4"></a>

* Used the above generated statistics, to separate classes into under represented classes and over represented classes.

#### - Training data set - under represented classes <a name="exploredata5"></a>

* Under representation data below stored as a dictionary.

* Key represents class ID, and value represents the number of times an image of this class ID needs to be generated, so that the total number of images of this class become close to the mean.

* For example, {0:3} in the data below, means that for an image of class 0, we will generate three fake images using this image during the data augmentation phase.

* Raw Data - {0: 3, 6: 1, 14: 0, 15: 0, 16: 1, 19: 3, 20: 1, 21: 1, 22: 1, 23: 0, 24: 2, 26: 0, 27: 2, 28: 0, 29: 2, 30: 1, 31: 0, 32: 2, 33: 0, 34: 1, 36: 1, 37: 3, 39: 1, 40: 1, 41: 2, 42: 2}

#### - Data augmentation <a name="exploredata6"></a>

* I augmented the under represented classes by randomly rotating them, then storing them to the disk for subsequent runs (pickle_data_for_analysis/X_train_fake_normalized.pickle and pickle_data_for_analysis/y_train_fake_normalized.pickle) . When I was trying to utilize this data later on, I was running into memory issues, so I haven't been able to use this data.

---

### - Preprocess the data as needed <a name="preprocessdata"></a>

* I decided to preprocess the data via normalization ( (pixel_intensity-128)/128 ) so as to help the CNN to achieve convergence earlier.

* Because this was a time consuming process, I saved these normalized images to disk as a pickle file, and for subsequent runs, used these saved normalized images.

* Visualized the data after normalization. Doing this helped me in catching an anomaly with my normalization technique.

---

### -  Design, train and test a model architecture <a name="designmodel"></a>

#### - Design a model architecture <a name="designmodel1"></a>

* Initial architecture chosen was the LeNet architecture because it already had good performance for image recognition tasks.

* Other options could have been ImageNet, AlexNet, GoogleNet, ResNet etc, but because I was getting good results with LeNet, so I persisted with it.

* However, inspite of tuning some of the hyper parameters, I was unable to exceed an accuracy of 89% because of which I chose to modify it.

* To achieve the current accuracy ( **93.94%** versus requirement of **93%** ) I made the following updates to the architecture :

	+ Increased the filter depth for convolution layers. This was done with the understanding the higher the filter depth, the more is the 'discriminating' capacity of the network. 
	+ Increased the dimensions of the fully connected layers. This was done to increase the 
	+ Added one more convolution layer with pooling. This was done with the understanding that the higher layers within a CNN are able to identify more complex shapes. 
	+ Introduced Dropout regularization ( with keep probability of 0.7 for training process, and 1 for validation and testing process.) This was done to prevent over fitting. Because I was already getting good results with dropout regularization, I decided to not pursue L2 regularization.
	+ Other than the above I experimented with adding one more convolution layer, but I did not see significant gain(s) from this and discarded this layer. Once approach that I could have tried, was to reduce the filter size for the initial layers from 5x5 to 3x3, and then add more filters in the latter layers.
	+ During my experimentation phase, I achieved a best validation accuracy of **96.5%**, though at the time, I did not know how to restore a saved model, and thus lost it.
	+ With this architecure in place, I was consistently hitting a validation accuracy of 90%+ within the first 7 to 10 epochs.
	+ The validation accuracy for the first epoch was around 40 - 50% which is good in the sense that we do not want our model to be super confident initially, but to gradually become more confident in it's predictions as training progresses.


* Defined common constants for hyper parameters in a single cell for quick visibility into my controlling knobs for the neural network.

* Defined convenience functions for : logging at different levels, printing memory usage, converting a class ID to a class name, rotate an image by an arbitrary angle, visualizing random samples from a given data set ( training, validation, test), storing preprocessed ( normalized ) data to the Disk as pickle file(s), making a sound ( to mark the completion of training ), showing progress indicators etc. These are generally useful functions, and I should be able to leverage them even outside of this specific project.

#### - Design a model architecture <a name="designmodel4"></a>

* **Model Layers**

	+ Define convenience methods to return a convolution layer ( with pooling and activation ), given the weights and biases.

	+ Define weights and biases as variables stored within the dictionary. The dimensions of these matrices were calculated with (F - P)/S + 1

	+ The final model architecture is as follows 
<BR>

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1 - 5x5   | 1x1 stride, valid padding	 					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 2 - 5x5   | 1x1 stride, valid padding	 					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32	 				|
| Convolution 3 - 5x5   | 1x1 stride, valid padding	 					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x256	 				|
| Flattened 	      	| outputs 1024					 				|
| Fully connected 1	    | outputs 240					 				|
| RELU					|												|
| Fully connected 2	    | outputs 168					 				|
| RELU					|												|
| Drop out      	    | outputs 168					 				|
| Output        	    | outputs 43					 				|

<BR>



#### - Train the model architecture <a name="designmodel2"></a>

* To help in the training process, I clearly identified all the hyper parameters in a code block.

* Identification of hyper parameters up front helped me in getting visibility into the 'knobs' which I had to control the training process.

* As expected, tweaking the hyper parameters, was the most difficult and
time consuming aspect of the project. An automated strategy for hyper parameter tuning would be an extra credit work for this project and extremely useful.

* Generally, I was able to achieve 90% accuracy on the validation set within the first 7 to 10 epochs. This served as my baseline to determine how well the model was 'converging', and whether to use or discard a hyper parameter combination.

* Once the epochs for training and validation had been executed, I wrote a JSON file which captured all the hyper parameters, any useful descriptions pertaining to this specific training process ( like preprocessing techniques), final validation accuracy achieved against the validation set, and also a per epoch capture of validation accuracy which was useful to see how the training was progressing. 

* I chose a JSON format ( over Pickle ), because it was easy to view it outside of the Python environment. 

#### - Test the model architecture <a name="designmodel3"></a>

* After I achieved 93.94% accuracy on the validation data set, I tested the same model against the test data set, and achieved a prediction accuracy of 91.7 %.

#### - A note on using Amazon GPUs for training <a name="designmodel5"></a>

* I completed the setup needed to execute the notebook on Amazon GPUs.

* I was also able to get a few test run(s) of the notebook on the GPU.

* However, early on, I suffered from quite a few random Kernel shutdown(s) and dead kernel(s), and the environment appeared to be unstable. 

* Because I was already getting 93.94% accuracy on my local workstation, within a reasonable time frame, I decided to pursue debugging of the GPU instance, though it could have been a good learning exercise.

* Subsequently, I terminated my EC2 GPU instance, and also removed the EBS storage attached to that instance, to prevent any unnecessary charges.

#### - Evaluating prediction failures in training data set <a name="designmodel6"></a>

* I added an 'inaccuracy' operation to Tensorflow, and with that was able to get the classes for which the model was failing the most within the training data set.

* Top three classes in the training data set for which the model prediction had the highest failure rate are (in descending order of failure rate)
<BR>


| Sign Name        				|     Class Id	       					| 
|:-----------------------------:|:-------------------------------------:| 
| End of speed limit (80km/h)   | 6	 									|
| Speed limit (100km/h) 		| 7	 									|
| Beware of ice/snow   			| 30	 								|

<BR>

#### - Evaluating prediction failures in validation data set <a name="designmodel7"></a>

* I added an 'inaccuracy' operation to Tensorflow, and with that was able to get the classes for which the model was failing the most within the validation data set.

* Top three classes in the validation data set for which the model prediction had the highest failure rate are (in descending order of failure rate)
<BR>


| Sign Name        				|     Class Id	       					| 
|:-----------------------------:|:-------------------------------------:| 
| Road narrows on the right   	| 24	 								|
| Dangerous curve to the right 	| 20	 								|
| Vehicles over 3.5 metric tons prohibited  	| 1						|

<BR>

#### - Evaluating prediction failures in testing data set <a name="designmodel8"></a>

* I added an 'inaccuracy' operation to Tensorflow, and with that was able to get the classes for which the model was failing the most within the testing data set.

* Top three classes in the testing data set for which the model prediction had the highest failure rate are - Pedestrians - Class ID : 27, Road narrows on the right - Class ID : 24, Dangerous curve to the right - Class ID : 20.

* Top three classes in the testing data set for which the model prediction had the highest failure rate are (in descending order of failure rate)
<BR>


| Sign Name        				|     Class Id	       					| 
|:-----------------------------:|:-------------------------------------:| 
| Pedestrians   				| 27	 								|
| Road narrows on the right 	| 24	 								|
| VDangerous curve to the right prohibited  	| 20					|

<BR>

#### - Final Results <a name="designmodel9"></a>

* The final results are : 
<BR>

| Data Set         		|     Accuracy	      		  					| 
|:---------------------:|:---------------------------------------------:| 
| Training         		| 	99.8%				   						| 
| Validation   			| 	93.94% 										|
| Test					|	91.7%										|

<BR>

* Because validation accuracy is greater than training accuracy, so currently the model is underfitted.

---

### - Performance against new images <a name="newimageperformance"></a>

#### - Acquire new German traffic sign images <a name="newimageperformance1"></a>

* New images "German traffic signs" as sourced from the internet

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

* The above images had dimensions of 100x100x4, so I resized them to 32x32, and discarded the alpha channel to get a depth of 3.

#### - Performance against new images <a name="newimageperformance2"></a>

* Achieved 60% success rate against new images.

#### - Misclassified images <a name="newimageperformance3"></a>

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]


#### - Output top-5 softmax predictions for these images <a name="newimageperformance4"></a>

* Softmax probabilities are a measure of how certain a model is, in it's predictions.

* **Misclassified image 1 - 27.gif**

![alt text][image13]

* Softmax predicted class labels for the above image

	+ [34 37 38 18 26]

* Predicted labels for the above image

	+ ['Turn left ahead' 'Go straight or left' 'Keep right'
  'General caution' 'Traffic signals']

* **Misclassified image 2 - 5.gif**

![alt text][image14]

* Softmax predicted class labels for the above image

	+ [10 42 9 23 20]

* Predicted labels for the above image

	+ ['No passing for vehicles over 3.5 metric ' 'End of no passing by vehicles over 3.5 m' 'No passing' 'Slippery road' 'Dangerous curve to the right']

* **Misclassified image 3 - unknown.gif**

![alt text][image15]

* Softmax predicted class labels for the above image

	+ [38 34 37 26 18]

* Predicted labels for the above image

	+ ['Keep right' 'Turn left ahead' 'Go straight or left'
  'Traffic signals' 'General caution']

* **Misclassified image 4 - unknown2.gif**

![alt text][image16]

* Softmax predicted class labels for the above image

	+ [ 4 21 40  1  5]

* Predicted labels for the above image

	+ ['Speed limit (70km/h)' 'Double curve' 'Roundabout mandatory'
  'Speed limit (30km/h)' 'Speed limit (80km/h)']

---


### - Retrospective <a name="retro"></a>

#### - Retrospective - what went well <a name="retro1"></a>

* Defining Convenience functions up-front helped with using them from different contexts. Additionally, these functions are general purpose, and can be used for other projects / subsequent assignments as well.

* Defining convenience functions and testing them right away also helped in reducing surprises.

* Identifying hyper parameters, up front gave me a good visibility into the knobs which I had for tuning.

* Became fairly comfortable with designing convolution layer(s) with different filter sizes, strides and depths.

* Logging at different level(s) helped me in getting the right level of information, and the right time. During development of a new feature, I debugLogged anything that needed to be logged, and once dev / testing on that feature was complete, I changed it to infoLog.

* For generated model(s), save them only if their validation accuracy improves.

* For generated model(s), save their validation accuracy in the file name, so that you have quick visibility into that information.

#### - Retrospective - areas for improvement, and learning <a name="retro2"></a>

* I struggled a bit, with restoring saved Tensorflow model(s). Because I was able to successfully complete the LeNet Lab on my own, I did not review the solution ( where restoration code was provided ). At one time, I had achieved a validation performance of 96.5% but at the time I did not know how to restore that model, hence I 'lost' that model over subsequent runs. **Learning** : Review the solution, even if you succeed in doing things on your own because you might still get to learn something new.

* The images loaded from the data set had a dtype of unit8, because of which when I tried to normalize and save them, the pixel intensities became integers, and so therefore always either 0 or 1. This naturally led to extremely low prediction accuracies. When I visualized the output images, it helped me detect the anomaly and then fix it. **Learning** : For image manipulation processes, visualize the output immediately after the manipulation to check if it looks as expected.

* Not necessarily an area for improvement, but more of familiarization with the Jupyter notebooks was that often I ran into Memory issues after running the notebook for a few hours. Also, in certain situations, the kernel died randomly. **Learning** : If you are running out of memory issues, it might be because you have chosen a large batch size. To mitigate this, reduce the batch size, terminate all unnecessary applicatins, terminate and restart your browser, and Docker and the Jupyter notebook.

* The process of training over multiple epochs took some time, and it took me some time to become patient and accept it as part of the dev workflow. **Learning** : Training a model is a time consuming process, so patience, coffee and liberal usage of GPUs is recommended ! Also, save good models aggresively, so that you can always restore them later if something goes wrong, and therefore you can 'save' all the effort that has been invested thus far into training the model.

* I am still not as comfortable with TensorFlow model save and restore, as I would like to be. **Learning** : Continue exploring TF model save and restore.

---

### - Additional exploration <a name="additionalexploration"></a>

* Need to find out what the state of the art for this problem is, and to tweak the network to achieve the same.

* In my analysis of under represented classes, I did the analysis for the training data set, but perhaps I also need to do the same for the validation data set. 

* A top level hyper parameter optimizer, which tweak(s) a hyper parameter, evaluates whether the model is improving ( e.g. in 10 epochs ), and then prints results of this evaluation to a human readable format, and then slowly does this for all the hyper parameters. This will automate a lot of the mundane task(s) around hyper parameter optimization.

* I was a bit conservative with the model, in the sense that I took the LeNet model and tweaked it to achieve the necessary performance. An even better approach would be start off with other model(s) like GoogleNet, or to start building a model from scratch and to evaluate it at every stage. 

* Understand and use TensorFlow Summary writer to save training summaries for a session.

* Any additional tuning items like – L2, deeper CNN, other optimizers, other pooling strategies, preprocessing like greyscaling. 

* Explore visualization samples - The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

* Do visualization of activation.

* Do visualization of prediction accuracy for validation data

* Do visualization of softmax probabilities for a particular new image

* Feature co ordinates within the image have been provided, so we could extract and use 
this information in some form.

* Implement Learning Rate Decay if accuracy does not improve over x-iterations. Also, need to understand how the Adam Optimizer implements this decay implicitly.

* Review ConvNetJS to check out how they implement various Data Visualizations in real time and see if you can incorporate those.

* For example using ConvNetJS, initiate a conv net training process, and see how the visualizations update in real time.

* Write a blog post on saving a model, and restoring it with Tensorflow.

---

### - Additional Reading <a name="additionalresources"></a>

#### - Additional Reading - CNN Architectures <a name="additionalresources1"></a>

* https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
* http://slazebni.cs.illinois.edu/spring17/lec04_advanced_cnn.pdf
* https://www.tensorflow.org/tutorials/image_recognition
* https://culurciello.github.io/tech/2016/06/04/nets.html

#### - Additional Reading - Tensorflow Best Practices <a name="additionalresources2"></a>

* https://github.com/aicodes/tf-bestpractice
* https://www.tensorflow.org/performance/performance_guide#best_practices
* https://indico.io/blog/the-good-bad-ugly-of-tensorflow/
* https://www.hakkalabs.co/articles/fabrizio
* http://web.stanford.edu/class/cs20si/syllabus.html

---

### - (Optional) Visualization of Neural network activation  <a name="activationvisualization"></a>

* Ran into issues with this section, so I will pursue this later.
