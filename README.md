# CS490-Deep-Learn-Project

## Quick Instruction
1. Install NumPy, tensorflow and OpenCV if you have not done yet
2. Run <a href="https://github.com/JakeACross/CS490-Deep-Learn-Project/blob/main/OpenCV%20FR%20Dataset%20Generation%20v2.ipynb">OpenCV FR Dataset Generation v2.ipynb</a> file to take pictures and save it to 'New Face' folder.
3. Download weight data from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
4. Add a path of the data to <a href="https://github.com/JakeACross/CS490-Deep-Learn-Project/blob/main/Main.ipynb">Main.ipynb</a> and run
5. Press 'q' to finish it 

## Program Descriptions


### FR Encryption Example:
Inside of the FR Encryption Example folder, there is a python program called: main.py. Run this to see
an example of encrypting a text file. This will be used as a proof of concept for the security part of
the project.

### FR Keras Pretrained Model Transfer Learning:
This model is our first model attempt for the Facial Recognition project. This model uses the VGG model
structure from keras to transfer learn the first five blocks of the typical vgg model structure. Once this
is done, the image dataset folders (train/test folders) are turned into the train/test data using
ImageDataGenerator to apply normalization and other preprocessing techniques. The model is fitted to the
train data. It has early stopping, and reduce learning rate on plateau callbacks applied, as well as the
test data as validation data. This model would be saved to the user's machine. 

### FR 

## Video Links:
Comparison between the two models:
https://youtu.be/ckxq1IgNzdY

Web application demonstration:
https://youtu.be/uWHrqNSWvug
