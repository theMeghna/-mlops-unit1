import logging
from flask import Flask, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Logging setup
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

app = Flask(__name__)

# Train model
data = load_iris()
model = RandomForestClassifier()
model.fit(data.data, data.target)

@app.route('/')
def home():
    logging.info("Home endpoint accessed")
    return "ML Model API Running"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['data']
    logging.info(f"Input received: {input_data}")

    prediction = model.predict([input_data])
    logging.info(f"Prediction: {prediction[0]}")

    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)