import logging
from flask import Flask, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(filename='app.log', level=logging.INFO)

app = Flask(__name__)

# Train model
data = load_iris()
model = RandomForestClassifier(n_estimators=50, max_depth=5)  # optimization
model.fit(data.data, data.target)

@app.route('/')
def home():
    return "ML Model API Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json['data']

        # 🔒 Security: input validation
        if not isinstance(input_data, list) or len(input_data) != 4:
            return jsonify({'error': 'Invalid input format'})

        # 🔒 Prevent extreme values (basic protection)
        if any(abs(x) > 100 for x in input_data):
            return jsonify({'error': 'Input out of range'})

        prediction = model.predict([input_data])
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        logging.error(str(e))
        return jsonify({'error': 'Something went wrong'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)