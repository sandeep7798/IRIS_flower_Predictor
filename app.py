from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load ML model
model = load_model("model.h5", compile=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        f1 = float(request.form["feature1"])
        f2 = float(request.form["feature2"])
        f3 = float(request.form["feature3"])
        f4 = float(request.form["feature4"])

        # Convert to numpy array
        features = np.array([[f1, f2, f3, f4]])

        # Prediction
        prediction = model.predict(features)

        result = prediction[0][0]

        return render_template("index.html", prediction_text=f"Prediction Result: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# Render deployment port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
