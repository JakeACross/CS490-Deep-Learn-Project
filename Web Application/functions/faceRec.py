""" This file comes from "Main.ipynb"
References:
Preprocess Image: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
VGG Code Reference: https://github.com/serengil/tensorflow-101/blob/master/python/deep-face-real-time.py
VGG Model Structure Reference: https://neurohive.io/en/popular-networks/vgg16/
Description: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
"""
# Libraries used:
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from scipy import spatial
from os import listdir

# Preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
# Devide the array by 127.5 and substract by 1
def preprocess_image(image):
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    
    img = preprocess_input(img)
    return img

# Create a CNN 
def loadMdl():
    # Download data from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing

    image_shapes = (224, 224, 3)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=image_shapes))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(4096, kernel_size=(7, 7), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(4096, kernel_size=(1, 1), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(2622, kernel_size=(1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    # Load pretrained weights for VGGFace
    model.load_weights('../../vgg_face_weights.h5')
    
    # Generate model with input and output
    vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face  # return the model

# Create the dictonary that saves the array of predictions for each images in the dataset
def setUpSource(model):
    # Initialize a file name and dictionary
    new_pictures_folder = 'Face Dataset'
    new_pictures = dict()

    # Iterate a folder ("Face Dataset") to folder ("User name") to user's 15 face images
    # String beocomes file path by listdir
    for folder_name in listdir(new_pictures_folder):
        if folder_name != '.DS_Store':  # for mac users
        
            folder_path = new_pictures_folder+'/'+folder_name  # create a path for the next folder to iterate
        
            pred = []  # initialize and reset a list
        
            for file_name in listdir(folder_path):
                file_name = folder_path+'/'+file_name  # create a path for the files
                face_img = load_img(file_name, target_size=(224, 224))  # load images by the path
                pred.append(model.predict(preprocess_image(face_img))[0,:])  # get values from each images by model.predict
            
            new_pictures[folder_name] = pred  # assign the list to dict

    return new_pictures


def faceRecognition(model):
    # A couple of variables are defined here. Color is the rgb value for black. user_face_matches
    # # and unauthorized_matches are counters for the results of the face verification.
    color = (0, 0, 0) 
    user_face_matches = 0
    unauthorized_matches = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    new_pictures = setUpSource(model)

    # The webcamera is opened up with OpenCV
    cap = cv2.VideoCapture(0)

    # This while loop will try to match detected faces with authorized users. If it succeeds, then the
    # counter goes up. If the detected face is an unauthorized user a different counter goes up. Once a
    # threshold of counters is met, the text file is updated
    while True:
        ret, img = cap.read()  # get a image from webcam capture
        faces = face_cascade.detectMultiScale(img, 1.3, 5)  # find faces in images
        # If the image capture is sucessful, this if statement is run
        if ret:
            for (x,y,w,h) in faces:  # x, y represents initial posions in a graph. w is width and h is height
                # If the width of the face is a relatively big, then it will be scanned with the model.
                if w > 130: 
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    user_detected = False

                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] # crop detected face
                    detected_face = cv2.resize(detected_face, (224, 224)) # resize to 224x224

                    img_pixels = preprocess_image(detected_face)
                    
                    # Assign the image extracted from webcam
                    captured_representation = model.predict(img_pixels)[0,:]
                    
                    for k in new_pictures:  # k is a key of dict which is user's name
                        folder = new_pictures[k]
                        for i in folder:
                            cosine_distance = spatial.distance.cosine(i, captured_representation)
                            cosine_similarity = 1 - cosine_distance
                            
                            # If cosine similarity is greater than 0.8, then consider that detected
                            # face as the user in folder k, and increment the counter
                            if(cosine_similarity > 0.80): 
                                cv2.putText(img, k, (int(x+w+5), int(y-5)), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
                                user_detected = True
                                user_face_matches += 1
                                break
                                
                    # If the detected image is not in user's face dataset, then display unauthorized
                    # and increment the unauthorized counter.
                    if(user_detected == False): 
                        cv2.putText(img, 'Unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        unauthorized_matches += 1

            # If the counter in authorized users is 10, then write 'Authorized' to teh text file.
            # As soon as, unauthorized users are detected 5 times, write 'Unauthorized' to teh text file
            if user_face_matches > 10:
                f = open('result.txt', 'w')
                f.write("Authorized")
                f.close()
                # camera.release()
                # cv2.destroyAllWindows()
            if unauthorized_matches > 5:
                f = open('result.txt', 'w')
                f.write("Unauthorized")
                f.close()
                # camera.release()
                # cv2.destroyAllWindows()
            
            # Displays the webcamera to the user
            cv2.imshow('img',img)

            ret, buffer = cv2.imencode('.jpg', img)  # encodes image formats into streaming data and stores it in-memory cache
            img = buffer.tobytes()  # convert it to bytes
            # Yield the encoded data, so it will come back until the face recognition is done
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
