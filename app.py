# app.py - V2.2 with Single-Day Search

import os
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import traceback
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
AERODATABOX_API_KEY = os.getenv("AERODATABOX_API_KEY")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # ... (this class is unchanged)
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
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/get-flights', methods=['POST'])
def get_flights():
    # This endpoint is now back to fetching flights for a single date.
    if not AERODATABOX_API_KEY:
        return jsonify({'error': 'API key for flight data is not configured.'}), 500

    try:
        search_params = request.get_json()
        origin = search_params.get('origin')
        destination = search_params.get('destination')
        flight_date = search_params.get('date')

        print(f"--> Received flight search for: {origin} -> {destination} on {flight_date}")

        if not all([origin, destination, flight_date]):
            return jsonify({'error': 'Missing origin, destination, or date.'}), 400

        url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{origin}/{flight_date}"
        querystring = {"withLeg":"true", "direction":"Departure", "withCancelled":"false", "withCodeshared":"true", "withCargo":"false", "withPrivate":"false"}
        headers = {
            "X-RapidAPI-Key": AERODATABOX_API_KEY,
            "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
        }

        api_response = requests.get(url, headers=headers, params=querystring)
        api_response.raise_for_status() 
        
        departures = api_response.json().get('departures', [])
        available_flights = []
        for flight in departures:
            if flight.get('arrival', {}).get('airport', {}).get('iata') == destination:
                available_flights.append({
                    'airline': flight['airline']['name'],
                    'number': flight['number'],
                    'departureTime': flight['departure']['scheduledTimeLocal'].split('+')[0],
                    'arrivalTime': flight['arrival']['scheduledTimeLocal'].split('+')[0],
                    'aircraft': flight.get('aircraft', {}).get('model', 'N/A'),
                    'flightDate': flight_date # The date is now the same for all results
                })
        
        print(f"<-- Found {len(available_flights)} matching flights.")
        return jsonify({'flights': available_flights})

    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            print(f"<-- No schedules found for {flight_date}.")
            return jsonify({'flights': []})
        else:
            return jsonify({'error': f'Flight API error: {http_err.response.text}'}), http_err.response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# --- (Your /get-premium endpoint is unchanged) ---
@app.route('/get-premium', methods=['POST', 'OPTIONS'])
def get_premium():
    # ... (This function is unchanged)
    if request.method == 'OPTIONS': return '', 204
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
        response_body = {'predicted_risk': prediction_probability, 'premium': user_premium, 'payout': payout}
        json_response = json.dumps(response_body, cls=NumpyEncoder)
        return Response(json_response, mimetype='application/json', status=200)
    except Exception as e:
        error_details = {'error': str(e), 'traceback': traceback.format_exc()}
        return Response(json.dumps(error_details), mimetype='application/json', status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)














# # app.py - FINAL CORRECTED VERSION
# import json
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import joblib
# import traceback # <--- Add this import for detailed error logging
# from flask import Flask, request, Response
# from flask_cors import CORS

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer): return int(obj)
#         if isinstance(obj, np.floating): return float(obj)
#         if isinstance(obj, np.ndarray): return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)

# print("Loading model...")
# model = tf.keras.models.load_model('flight_risk_model.h5')
# print("Model loaded successfully.")

# print("Loading preprocessor...")
# preprocessor = joblib.load('preprocessor.joblib')
# print("Preprocessor loaded successfully.")

# app = Flask(__name__)
# CORS(app, resources={r"/get-premium": {"origins": "*"}})

# @app.route('/get-premium', methods=['POST', 'OPTIONS'])
# def get_premium():
#     if request.method == 'OPTIONS':
#         return '', 204

#     try:
#         input_data_json = request.get_json()
#         user_premium = float(input_data_json.get('premium', 100))
#         flight_data = pd.DataFrame([input_data_json])

#         input_data_processed = preprocessor.transform(flight_data)
#         prediction_probability = model.predict(input_data_processed)[0][0]
        
#         margin = 0.15 
#         payout = 0
#         if prediction_probability > 0:
#              payout = (user_premium / prediction_probability) * (1 - margin)
        
#         response_body = {
#             'predicted_risk': prediction_probability,
#             'premium': user_premium,
#             'payout': payout
#         }
#         json_response = json.dumps(response_body, cls=NumpyEncoder)
#         return Response(json_response, mimetype='application/json', status=200)

#     except Exception as e:
#         # Provide a detailed error if something goes wrong on the backend
#         error_details = {'error': str(e), 'traceback': traceback.format_exc()}
#         return Response(json.dumps(error_details), mimetype='application/json', status=500)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
