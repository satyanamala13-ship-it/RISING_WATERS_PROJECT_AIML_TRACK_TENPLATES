from flask import Flask, render_template, request, redirect
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("transform.save")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temp"])
    humidity = float(request.form["humidity"])
    cloud = float(request.form["cloud"])
    annual = float(request.form["annual"])
    jan_feb = float(request.form["jan_feb"])
    mar_may = float(request.form["mar_may"])
    jun_sep = float(request.form["jun_sep"])
    oct_dec = float(request.form["oct_dec"])
    avgjune = float(request.form["avgjune"])
    sub = float(request.form["sub"])
    data = np.array([[temp,
                      humidity,
                      cloud,
                      annual,
                      jan_feb,
                      mar_may,
                      jun_sep,
                      oct_dec,
                      avgjune,
                      sub]])
    data = scaler.transform(data)
    prediction = model.predict(data)
    print("Prediction:", prediction)
    print("probability:",model.predict_proba(data))
    print("Input:",data)
    if prediction[0] ==1:
        return redirect("/chance")
    else:
        return redirect("/no-chance")
@app.route("/chance")
def chance():
    return render_template("chance.html")
@app.route("/no-chance")
def no_chance():
    return render_template("no_chance.html")
if __name__ == "__main__":
    app.run(debug=True)