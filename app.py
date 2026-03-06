from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os

# Hide TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load iris dataset
iris = load_iris()

# Load trained model
model = load_model("model.keras")

# Fit scaler
scaler = StandardScaler()
scaler.fit(iris.data)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            data = scaler.transform(data)

            pred = model.predict(data, verbose=0)
            predicted_class = np.argmax(pred)

            prediction = iris.target_names[predicted_class]

        except:
            prediction = "Invalid Input"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
