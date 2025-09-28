# app.py - FINAL CORRECTED VERSION
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import traceback # <--- Add this import for detailed error logging
from flask import Flask, request, Response
from flask_cors import CORS

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

print("Loading model...")
model = tf.keras.models.load_model('flight_risk_model.h5')
print("Model loaded successfully.")

print("Loading preprocessor...")
preprocessor = joblib.load('preprocessor.joblib')
print("Preprocessor loaded successfully.")

app = Flask(__name__)
CORS(app, resources={r"/get-premium": {"origins": "*"}})

@app.route('/get-premium', methods=['POST', 'OPTIONS'])
def get_premium():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        input_data_json = request.get_json()
        user_premium = float(input_data_json.get('premium', 100))
        flight_data = pd.DataFrame([input_data_json])

        input_data_processed = preprocessor.transform(flight_data)
        prediction_probability = model.predict(input_data_processed)[0][0]
        
        margin = 0.15 
        payout = 0
        if prediction_probability > 0:
             payout = (user_premium / prediction_probability) * (1 - margin)
        
        response_body = {
            'predicted_risk': prediction_probability,
            'premium': user_premium,
            'payout': payout
        }
        json_response = json.dumps(response_body, cls=NumpyEncoder)
        return Response(json_response, mimetype='application/json', status=200)

    except Exception as e:
        # Provide a detailed error if something goes wrong on the backend
        error_details = {'error': str(e), 'traceback': traceback.format_exc()}
        return Response(json.dumps(error_details), mimetype='application/json', status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
