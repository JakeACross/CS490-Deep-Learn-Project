from flask import Flask, render_template, request, Response
from functions.makeDS import *
from functions.faceRec import *
from os import listdir



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/", methods=["POST"])
def makeMdl():
    user_name = request.form['name']
    for folder_name in listdir('Face Dataset'):
        if folder_name == user_name: 
            return render_template('faceRec.html')
    face_dataset_generator(user_name)
    return render_template('faceRec.html')
    
@app.route("/faceRec")
def faceRec():
    f = open('result.txt', 'w+')
    f.close()
    model = loadMdl()
    func = faceRecognition(model)
    return Response(func,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/logIn")
def logIn():
    f = open('result.txt', 'r')
    result = f.readline()
    if result == "Authorized":
        return render_template('user.html')
    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    app.run(debug=True)


''' Hold shift and click reload button to update css -> https://www.pythonanywhere.com/forums/topic/7425/ '''