from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

app = Flask(__name__)

iris = load_iris()

model = load_model("iris_model.h5")

scaler = StandardScaler()
scaler.fit(iris.data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = scaler.transform([features])
    prediction = model.predict(final)
    output = np.argmax(prediction)

    return render_template('index.html', prediction_text="Predicted class: {}".format(output))

if __name__ == "__main__":
    app.run()
