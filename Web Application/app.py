from flask import Flask, render_template, request, Response, redirect
from functions.makeDS import *
from functions.faceRec import *
from os import listdir



app = Flask(__name__)

# Home Page: just display the html file
@app.route("/")
def home():
    return render_template("home.html")

# Home Page: recieve input text and if the name is not in the dataset, it will take pictures then start the face detection
# Otherwise, it will start the face detection
@app.route("/", methods=["POST"])
def makeMdl():
    user_name = request.form['name']  # get 'name' from the form on html
    for folder_name in listdir('Face Dataset'):
        if folder_name == user_name: 
            return render_template('faceRec.html')
    face_dataset_generator(user_name)
    return render_template('faceRec.html')

# Face Recognition Page: display the result of video capture on the web page    
@app.route("/faceRec")
def faceRec():
    f = open('result.txt', 'w+')  # delete the text on the text file
    f.close()
    model = loadMdl()  # create a CNN model with weights
    func = faceRecognition(model)  # open video captures
    return Response(func,
                    mimetype='multipart/x-mixed-replace; boundary=frame')  # send all video infomation to the link on the home html

# Login Page: display the simple html file if users login successfully.
# Otherwise, just redirect to the home page
@app.route("/logIn")
def logIn():
    # Open the text file and read to see the user's status
    f = open('result.txt', 'r')
    result = f.readline()

    if result == "Authorized":
        return render_template('user.html')  # load the user page
    else:
        return redirect('/')  # redirect to the home page
    
if __name__ == "__main__":
    app.run(debug=True)


''' Hold shift and click reload button to update css -> https://www.pythonanywhere.com/forums/topic/7425/ '''