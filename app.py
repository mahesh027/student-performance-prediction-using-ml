from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
import pickle

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("login.html")



@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    
    if username == "savu p sudheep" and password == "canva@12345":
        # Redirect to home.html if login is successful
        return render_template("home.html")
    else:
        # Render an error message or redirect back to login page
        return render_template("login.html", error="Invalid credentials")



@app.route("/result",methods=['POST','GET'])
def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    Medu= int(request.form['Medu'])
    Fedu = int(request.form['Fedu'])
    G1=float(request.form['G1'])
    G2=float(request.form['G2'])
    traveltime = int(request.form['traveltime'])
    studytime = int(request.form['studytime'])
    failures = int(request.form['failures'])
    schoolsup = int(request.form['schoolsup'])
    internet = int(request.form['internet'])
    goout = int(request.form['goout'])
    health = int(request.form['health'])
    absences = int(request.form['absences'])
    
    

    x=np.array([gender,age,Medu,Fedu,G1,G2,traveltime,studytime,failures,schoolsup,internet,goout,health,absences]).reshape(1,-1)

    scaler_path = os.path.join('models\scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('models\lr.sav')
    lr=joblib.load(model_path)

    Y_pred=lr.predict(x)

    
    if Y_pred==0:
        return render_template('nopass.html')
    else:
        return render_template('pass.html')

if __name__=="__main__":
    app.run(debug=True,port=7384)