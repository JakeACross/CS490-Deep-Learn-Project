# CS490-Deep-Learn-Project

## Quick Instruction
1. Install NumPy, tensorflow and OpenCV if you have not done yet
2. Run <a href="https://github.com/JakeACross/CS490-Deep-Learn-Project/blob/main/OpenCV%20FR%20Dataset%20Generation%20v2.ipynb">OpenCV FR Dataset Generation v2.ipynb</a> file to take pictures and save it to 'New Face' folder.
3. Download weight data from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
4. Add a path of the data to <a href="https://github.com/JakeACross/CS490-Deep-Learn-Project/blob/main/Main.ipynb">Main.ipynb</a> and run
5. Press 'q' to finish it 

## Program Descriptions

### Main Jupyter Notebook File:
This is the main file to run after the dataset the user wants to use is created. This file not only
creates the second model for facial recognition, but also contains the OpenCV facial verification and
an proof of concept of implementing security using this system. 

Firstly, to discuss the model. It utilizes the VGG Face dataset to train the model. 
This was done to try to fix the main issues we noticed with the first model (FR TL Model). (See paragraph [1]).
The structure of this model is very similar to a standard VGG16 model with the first five blocks of layers
mirroring a VGG16 model. The final block of layers results in an output equal to the number of facial
structures in the VGG Face dataset, which was 2622. Doing this creates a opportunity for the model to have
more options to pick from when predicting faces. So, instead of just being able to predict the facial structures
of the people within the 'New Faces' dataset, it assigns the user faces to faces within the VGG Face dataset.
This effectively eliminated the issue of unauthorized users being predicted as authorized users within the system.

After the model is loaded, then the program opens up a web camera for OpenCV to detect the user's face. It then
tries to predict which face it is currently detecting. If the user is an authorized user, then their name is 
displayed next to them. If the user is an unauthorized user, then 'Unauthorized user' is displayed above them.
  
Lastly, a simple security measure was implemented as a proof of concept. If the model determines the current
user is an authorized user, then a text file is decoded and outputted to the user. However, if the model
determines the user is an unauthorized user, then the text file remains encrypted and outputted as a bunch of
random characters that the user would not be able to decipher.

[1]The first model (from FR Keras program) was predicting authorized user faces pretty well, but struggled with 
predicting unknown faces. There are a number of reasons why this might have been an issue. Even though we 
took lots of measures to prevent this, the first model could have been due to overfitting. This issue could have 
also been due to the relatively low number of images that were trained to each face. Also, the original model
could only predict outcomes equal to the number of people within the dataset. We attempted to implement a threshold
for the predictions, but were still left issues. Unfortunately, we ran out of time to try to come up 
with more solutions for the first model, so we created a second model within this main file using a much
larger dataset.

***Note: A new user only needs to run the OpenCV FR Dataset Generation file to see this model in action. Alternatively, 
the user could also run the flask web application.***

### OpenCV FR Dataset Generation:
This program uses OpenCV to gather the images from the current user and saves it into the 'New Faces'
folder.

### FR Encryption Example:
Inside of the FR Encryption Example folder, there is a python program called: main.py. Run this to see
an example of encrypting a text file. This will be used as a proof of concept for the security part of
the project.

## These programs only apply to the first model, the VGG16 Transfer Learning model created with the
## 'New Faces' dataset.

### FR Keras Pretrained Model Transfer Learning:
This model is our first model attempt for the Facial Recognition project. This model uses the VGG model
structure from keras to transfer learn the first five blocks of the typical vgg model structure. Once this
is done, the image dataset folders (train/test folders) are turned into the train/test data using
ImageDataGenerator to apply normalization and other preprocessing techniques. The model is fitted to the
train data. It has early stopping, and reduce learning rate on plateau callbacks applied, as well as the
test data as validation data. This model would be saved to the user's machine. 

### FR Image Folders to Train_Test Folders:
This program simply breaks up a folder into train/val/test.

### FR TL Video Capture Example:
This program mirrors the main jupyter notebook file, but with the changes to allow the folders created during
the first model creation (FR Keras Pretrained Model Transfer Learning) to function like the main jupyter notebook

## Video Links:
Comparison between the two models:
https://youtu.be/ckxq1IgNzdY

Web application demonstration:
https://youtu.be/uWHrqNSWvug
