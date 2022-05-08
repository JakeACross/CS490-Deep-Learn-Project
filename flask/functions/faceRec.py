import numpy as np
import cv2

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from scipy import spatial

from os import listdir

def preprocess_image(image):
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    
    img = preprocess_input(img)
    return img

def loadMdl():
    # Create a CNN 
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

def faceRecognition(model):
    color = (0, 0, 0) # RGB value for black
    user_face_matches = 0
    unauthorized_matches = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    new_pictures = setUpSource(model)
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()  # get a image from webcam capture
        faces = face_cascade.detectMultiScale(img, 1.3, 5)  # find faces in images

        for (x,y,w,h) in faces:  # x, y represents initial posions in a graph. w is width and h is height
            if w > 130: 
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                user_detected = False

                detected_face = img[int(y):int(y+h), int(x):int(x+w)] # crop detected face
                detected_face = cv2.resize(detected_face, (224, 224)) # resize to 224x224

                img_pixels = preprocess_image(detected_face)
                
                # Assign the image extracted from webcam
                captured_representation = model.predict(img_pixels)[0,:]

                found = 0  # counter
                
                for k in new_pictures:  # k is a key of dict which is user's name
                    folder = new_pictures[k]
                    for i in folder:
                        cosine_distance = spatial.distance.cosine(i, captured_representation)
                        cosine_similarity = 1 - cosine_distance
                        # cosine_similarity = cosineSimilarity(i, captured_representation)
                        # If cosine distance is small, consider it is k (user's name)
                        if(cosine_similarity > 0.80): 
                            cv2.putText(img, k, (int(x+w+5), int(y-5)), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
                            user_detected = True
                            user_face_matches += 1
                            break

                # if found image is not in user's face dataset, display unknown
                if(user_detected == False): 
                    cv2.putText(img, 'Unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    unauthorized_matches += 1

        if user_face_matches > 30:
            f = open('result.txt', 'w')
            f.write("Authorized")
            f.close()
        if unauthorized_matches > 10:
            f = open('result.txt', 'w')
            f.write("Unauthorized")
            f.close()

        cv2.imshow('img',img)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
