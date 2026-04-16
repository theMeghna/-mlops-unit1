
from flask import Flask, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Train model
data = load_iris()
model = RandomForestClassifier()
model.fit(data.data, data.target)

@app.route('/')
def home():
    return "ML Model API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)