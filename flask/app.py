from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/makeMdl")
def makeMdl():
    return render_template("makeMdl.html")
    
@app.route("/faceRec")
def faceRec():
    return render_template("faceRec.html")
    
if __name__ == "__main__":
    app.run(debug=True)


''' Hold shift and click reload to update css -> https://www.pythonanywhere.com/forums/topic/7425/ '''