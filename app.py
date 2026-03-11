from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model

# Hide TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# Load trained model
model = load_model("model.keras", compile=False)

# Iris flower classes
flower_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        sepal_length = float(request.form["feature1"])
        sepal_width = float(request.form["feature2"])
        petal_length = float(request.form["feature3"])
        petal_width = float(request.form["feature4"])

        # Convert input to numpy array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Model prediction
        prediction = model.predict(input_data)

        # Get predicted class
        predicted_index = np.argmax(prediction)

        # Get flower name
        result = flower_names[predicted_index]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Iris Flower is: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
