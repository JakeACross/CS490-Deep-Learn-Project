import cv2
from pathlib import Path

def face_dataset_generator(name):
    image_count = 145
    image_path = '../New Faces'
    user_name = name
    Path('{}/{}'.format(image_path, user_name)).mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(0)

    while image_count <= 150:
        result, image = camera.read()

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
        
    camera.release()
    cv2.destroyAllWindows()

def createModel():
    pass