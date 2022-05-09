""" This file comes from "OpenCV FR Dataset Generation v2.ipynb" """
# Libraries necessary to use OpenCV and saving files to a specific path 
import cv2
from pathlib import Path

# This function utilizes OpenCV to gather face images for a facial recognition dataset. The parameter becomes the folder name
def face_dataset_generator(name):
    image_count = 1  # the image count is set to 1 it will be used to label the images in the while loop
    image_path = 'Face Dataset'  # the folder name of all folders
    user_name = name  # the file folder name
    Path('{}/{}'.format(image_path, user_name)).mkdir(parents=True, exist_ok=True)
    
    # This while loop will run until the program grabs and saves 15 cropped images to the specified path
    camera = cv2.VideoCapture(0)
    while image_count <= 15:
        # The camera is initialized here and captures an image
        result, image = camera.read()
        # If the image capture is sucessful, this if statement is run.
        if result:
            cv2.imshow('OpenCV Camera Test', image)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            detected_faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
        
        for (x, y, w, h) in detected_faces:
            # A rectangle is formed around the face and the face is cropped.
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cropped_face = image[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, dsize=(224, 224))
            # The cropped image is saved to the path and image counter is increased.
            cv2.imwrite(r'{}/{}/{}_{}.jpg'.format(image_path, user_name, user_name, image_count), cropped_face)
            image_count += 1

    # The camera is turned off and all OpenCV windows are destroyed.   
    camera.release()
    cv2.destroyAllWindows()