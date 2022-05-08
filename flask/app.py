from flask import Flask, render_template, request, redirect, Response
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
            return redirect('/faceRec')
    face_dataset_generator(user_name)
    return redirect('/faceRec')
    
@app.route("/faceRec")
def faceRec():
    model = loadMdl()
    func = faceRecognition('Jake Cross', model)
    return Response(func,
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    if func:
        return render_template("success.html")
    else:
        return redirect('/')
    
if __name__ == "__main__":
    app.run(debug=True)


''' Hold shift and click reload button to update css -> https://www.pythonanywhere.com/forums/topic/7425/ '''