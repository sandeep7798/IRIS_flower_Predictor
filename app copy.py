from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os

app = Flask(__name__)

# Load dataset
iris = load_iris()

# Load trained model
model = load_model("model.keras")

# Fit scaler
scaler = StandardScaler()
scaler.fit(iris.data)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == "POST":
        try:
            sl = float(request.form['sepal_length'])
            sw = float(request.form['sepal_width'])
            pl = float(request.form['petal_length'])
            pw = float(request.form['petal_width'])

            data = np.array([[sl, sw, pl, pw]])
            data = scaler.transform(data)

            pred = model.predict(data)
            predicted_class = np.argmax(pred)

            prediction = iris.target_names[predicted_class]

        except Exception as e:
            print("Error:", e)
            prediction = "Invalid Input"

    return render_template("index.html", prediction=prediction)


# For local testing and deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)