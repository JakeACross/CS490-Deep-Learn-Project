from flask import Flask, render_template, request, redirect
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
    for folder_name in listdir('../New Faces'):
        if folder_name == user_name: 
            return redirect('/faceRec')
    face_dataset_generator(user_name)
    return redirect('/faceRec')

    
@app.route("/faceRec")
def faceRec():
    model = loadMdl()

    return render_template("faceRec.html")
    
if __name__ == "__main__":
    app.run(debug=True)


''' Hold shift and click reload button to update css -> https://www.pythonanywhere.com/forums/topic/7425/ '''