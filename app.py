from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os

app = Flask(__name__)

# Load dataset (for labels only)
iris = load_iris()

# Load trained model
model = load_model("model.keras")

# IMPORTANT: Scaler must match training
scaler = StandardScaler()
scaler.fit(iris.data)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sl = float(request.form['sl'])
        sw = float(request.form['sw'])
        pl = float(request.form['pl'])
        pw = float(request.form['pw'])

        data = np.array([[sl, sw, pl, pw]])
        data = scaler.transform(data)

        prediction = model.predict(data, verbose=0)
        predicted_class = np.argmax(prediction)

        result = iris.target_names[predicted_class]

        return render_template("result.html", prediction=result)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", prediction="Invalid Input")


# Only needed for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT
    app.run(host="0.0.0.0", port=port, debug=False)
