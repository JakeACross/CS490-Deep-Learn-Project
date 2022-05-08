import numpy as np
import cv2

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from os import listdir

def preprocess_image(image):
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    
    img = preprocess_input(img)
    return img

def findCosineDistance(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, source))
    c = np.sum(np.multiply(test, test))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def loadMdl():
    # Create a CNN 
    # Download data from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    # Load pretrained weights for VGGFace
    model.load_weights('../../vgg_face_weights.h5')
    
    # Generate model with input and output
    vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face  # return the model

def setUpSource(model):
    # Initialize a file name and dictionary
    new_pictures_folder = 'Face Dataset'
    new_pictures = dict()

    # Iterate a folder ("New Faces") to folder ("User name") to user's 150 face images
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

def faceRecognition(name, model):
    color = (0,255,0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    new_pictures = setUpSource(model)
    result = 0
    
    cap = cv2.VideoCapture(0)

    for i in range(7):
        ret, img = cap.read()  # get a image from webcam capture
        faces = face_cascade.detectMultiScale(img, 1.3, 5)  # find faces in images

        for (x,y,w,h) in faces:  # x, y represents initial posions in a graph. w is width and h is height
            if w > 130: 

                detected_face = img[int(y):int(y+h), int(x):int(x+w)] # crop detected face
                detected_face = cv2.resize(detected_face, (224, 224)) # resize to 224x224

                img_pixels = preprocess_image(detected_face)
                
                # Assign the image extracted from webcam
                captured_representation = model.predict(img_pixels)[0,:]

                found = 0  # counter
                
                
                folder = new_pictures[name]
                for i in folder:
                    distance = findCosineDistance(i, captured_representation)
                        
                    # If cosine distance is small, consider it is k (user's name)
                    if(distance < 0.30): 
                        cv2.putText(img, name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        found = 1
                        result += 1
                        break

                # Connect face and text
                cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),color,1)
                cv2.line(img,(x+w,y-20),(x+w+10,y-20),color,1)
                
                # if found image is not in user's face dataset, display unknown
                if(found == 0): 
                    cv2.putText(img, 'Unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('img',img)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

    # Kill open cv things
    cap.release()
    cv2.destroyAllWindows()

    if result >= 6:
        return True
    else:
        return False

