# Self-Driving-Cars-Term1-Project2

**Udacity Self Driving Car - Term 1 - Project 2 - Traffic Sign Classifier - submission by Jai Lad **

**Objective** : 

* Given a set of German traffic signs, design and train a convolutional neural network ( CNN ) in Tensorflow, to achieve a validation accuracy of at least 93%.

* Explain the process of designing, validating and testing the CNN and the overall solution approach.

* Download new German traffic sign images from the internet, and explore how your trained model performs against these new images.

**Pre-Requisite** : 

* Term1 - starter kit
https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md

I followed the instructions for Docker image ( Mac ) because I preferred to have the same code easily deployable to my local workstation ( with CPU ) versus GPU ( Amazon ), and Docker allows for that flexibility. This was also done to gain more familiarity with the Docker environment, and also because the setup was fairly straightforward.

**Packages used to build the pipeline / CNN** : 

Once the above was installed I used the following packages to build the pipeline :

* Numpy - for matrix manipulations.
* TensorFlow - to design, train, validate and test the CNN.
* MatPlotLib - to generate visualizations.
* Pickle - to serialize / deserialize data to the Disk. 
* Json - to write training performance data to the Disk ( so that it was easily human consumable ).
* OpenCV - for image manipulations, like resizing, and converting a 4 channel image to a 3 channel image.
* Others - os, sys, time, copy, resource, scipy

**Tweakable parameters for the project** : 

* Epochs
* Batch Size
* Feature images mean 
* Feature images standard deviation 
* Learning Rate
* Type of activation function.
* Type of pooling.
* Data transormations - like rotation and preprocessing of data.
* Type of optimizer
* Regularization - Dropout

**Key File(s)** :

* [writeup.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/writeup.md) - Detailed project write-up.
* [Traffic_Sign_Classifier_submit/Traffic_Sign_Classifier.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_submit/Traffic_Sign_Classifier.md) - Fully executed Python notebook - Markdown format.
* [Traffic_Sign_Classifier_submit.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_submit.ipynb) - Fully executed Python notebook - iPynb format.
* [Traffic_Sign_Classifier_submit.html](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_submit.html) - Fully executed Python notebook - HTML format.
* [Traffic_Sign_Classifier_dev.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project2/blob/master/Traffic_Sign_Classifier_dev.ipynb) - The actual developmental Python notebook from which the above submissions were exported.
* checkpoint - Tensorflow Checkpoint.
* Files named - 'jltrafficclassifier9394' - This is the trained model with 93.94% validation accuracy.
* SideExperiments.ipynb - a small notebook which I used during the development process, to avoid 'polluting' the main dev notebook.
* signnames.csv - converts Class ID values to human readable descriptions.
* README.md - this file. :-)

**Key Folder(s)** :

* data_for_analysis - the folder which contains the summary of a training session. The file(s) contain data in JSON format, and can be opened by any text editor ( e.g. Sublime Text ) For example, the file '1492470140.0814645' contains the details of the training session used for this submission.

* NewImages - The folder which contains new German traffic acquired from the web.

* pickle_data_for_analysis - The folder to which the program saves normalized images ( and generated fake data images for under represented classes), so that they can be read from here for subsequent runs, therefore saving time.

* Traffic_Sign_Classifier - the folder which contains the fully executed version of the notebook, in markdown format - Traffic_Sign_Classifier.md

* OlderNotebooks - Backups of previously exported versions of the dev notebook. 
