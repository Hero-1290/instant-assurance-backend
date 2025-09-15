import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask import Flask, request, Response
from flask_cors import CORS

# A custom JSON encoder for NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- GLOBAL SCOPE: Load models once on startup ---
print("Loading model...")
model = tf.keras.models.load_model('flight_risk_model.h5')
print("Model loaded successfully.")

print("Loading preprocessor...")
preprocessor = joblib.load('preprocessor.joblib')
print("Preprocessor loaded successfully.")

# --- FLASK APP DEFINITION ---
app = Flask(__name__)
# Explicitly configure CORS for our specific endpoint to avoid browser issues
CORS(app, resources={r"/get-premium": {"origins": "*"}})

@app.route('/get-premium', methods=['POST'])
def get_premium():
    try:
        input_data_json = request.get_json()
        input_data = pd.DataFrame([input_data_json])
        input_data_processed = preprocessor.transform(input_data)
        prediction_probability = model.predict(input_data_processed)[0][0]
        
        # This is the risk-adjusted payout logic we finalized
        payout = 0
        premium = 0 # You would get the user's desired premium from the JSON input
        # Example payout calculation:
        # premium = input_data_json['premium']
        # margin = 0.15 
        # payout = (premium / prediction_probability) * (1 - margin)
        
        response_body = {
            'predicted_risk': prediction_probability,
            'premium': premium, # Send back the user's premium
            'payout': payout # Send back the calculated payout
        }
        json_response = json.dumps(response_body, cls=NumpyEncoder)
        return Response(json_response, mimetype='application/json', status=200)

    except Exception as e:
        return Response(json.dumps({'error': str(e)}), mimetype='application/json', status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
