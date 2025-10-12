# app.py

import os
import requests
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime, timedelta
from keras.models import load_model

# --- (Sections 1, 2, and 3 are unchanged) ---
load_dotenv()
AERODATABOX_API_KEY = os.getenv("AERODATABOX_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

print("🔄 Loading model and preprocessor...")
model = load_model('flight_risk_model_final.h5')
preprocessor = joblib.load('preprocessor_final.joblib')
print("✅ Model and Preprocessor loaded successfully.")

app = Flask(__name__)
CORS(app)

def get_weather_forecast(iata_code):
    coords = {'ATL': (33.64, -84.42), 'LAX': (33.94, -118.40), 'JFK': (40.64, -73.78)}.get(iata_code)
    if not coords: return {'Temp': 22, 'Wind': 5, 'Condition': 'Clear'}
    lat, lon = coords
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    response = requests.get(url)
    response.raise_for_status()
    weather = response.json()['list'][8]
    return {'Temp': weather['main']['temp'], 'Wind': weather['wind']['speed'], 'Condition': weather['weather'][0]['main']}

@app.route('/search-flights', methods=['POST'])
def search_flights():
    try:
        data = request.get_json()
        origin = data['origin']
        destination = data['destination']
        flight_date = data['date']

        url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{origin}/{flight_date}"
        params = {"withLeg": "true", "direction": "Departure"}
        headers = { "X-RapidAPI-Key": AERODATABOX_API_KEY, "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com" }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        departures = response.json().get('departures', [])
        # ... (Your original parsing logic would go here)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("🚧 Live API failed (404). Returning mock flight data for testing.")
            base_time = datetime.fromisoformat(f"{flight_date}T09:00:00")
            mock_flights = [
                {'Airline': 'DL', 'AirportFrom': origin, 'AirportTo': destination, 'flight_number': 'DL123', 'departureTime': base_time.isoformat(), 'duration': 245},
                {'Airline': 'AA', 'AirportFrom': origin, 'AirportTo': destination, 'flight_number': 'AA456', 'departureTime': (base_time + timedelta(hours=2)).isoformat(), 'duration': 250},
                {'Airline': 'UA', 'AirportFrom': origin, 'AirportTo': destination, 'flight_number': 'UA789', 'departureTime': (base_time + timedelta(hours=4)).isoformat(), 'duration': 240}
            ]
            return jsonify({'flights': mock_flights})
        else:
            return jsonify({'error': f"Flight API error: {str(e)}"}), 500
        
    except Exception as e:
        return jsonify({'error': f"An internal server error occurred: {str(e)}"}), 500

# --- START OF FIX ---
@app.route('/get-quote', methods=['POST'])
def get_quote():
    try:
        data = request.get_json()
        departure_dt = datetime.fromisoformat(data['departureTime'])
        origin_weather = get_weather_forecast(data['AirportFrom'])
        dest_weather = get_weather_forecast(data['AirportTo'])
        
        model_input = {
            'Airline': data['Airline'], 'AirportFrom': data['AirportFrom'], 'AirportTo': data['AirportTo'],
            'DayOfWeek': departure_dt.weekday() + 1, 'Time': departure_dt.hour * 60 + departure_dt.minute,
            'Length': data['duration'], 'WeatherFrom_Temp': origin_weather['Temp'],
            'WeatherFrom_Wind': origin_weather['Wind'], 'WeatherFrom_Condition': origin_weather['Condition'],
            'WeatherTo_Temp': dest_weather['Temp'], 'WeatherTo_Wind': dest_weather['Wind'],
            'WeatherTo_Condition': dest_weather['Condition']
        }
        
        input_df = pd.DataFrame([model_input])
        processed_input = preprocessor.transform(input_df)
        
        # model.predict() returns a numpy float32, which is not JSON serializable
        risk_numpy = model.predict(processed_input)[0][0]
        
        # Convert the numpy types to standard Python floats
        risk = float(risk_numpy)
        user_premium = float(data.get('premium', 100))
        margin = 0.15
        payout = float((user_premium / risk) * (1 - margin) if risk > 0 else 0)
        
        return jsonify({ 
            'predicted_risk': risk, 
            'premium': user_premium, 
            'payout': payout 
        })
    except Exception as e:
        return jsonify({'error': f"Quote calculation failed: {str(e)}"}), 500
# --- END OF FIX ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)










# import os
# import requests
# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# from datetime import datetime

# # --- 1. Load Environment Variables and API Keys ---
# load_dotenv()
# AERODATABOX_API_KEY = os.getenv("AERODATABOX_API_KEY")
# OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY") # You will need to add this to your .env file

# # --- 2. Load ML Model and Preprocessor ---
# print("🔄 Loading model...")
# # Ensure you are using the final model and preprocessor trained on the weather data
# model = tf.keras.models.load_model('flight_risk_model_final.h5') 
# preprocessor = joblib.load('preprocessor_final.joblib')
# print("✅ Model and Preprocessor loaded successfully.")

# app = Flask(__name__)
# CORS(app) # Allow all origins for simplicity

# # --- 3. Helper Functions for External APIs ---

# def get_flight_details(origin_iata, flight_date, flight_number):
#     """Fetches detailed flight info (Time, Length) from AeroDataBox."""
#     url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{origin_iata}/{flight_date}"
#     params = {"withLeg": "true", "direction": "Departure"}
#     headers = {
#         "X-RapidAPI-Key": AERODATABOX_API_KEY,
#         "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
#     }
#     response = requests.get(url, headers=headers, params=params)
#     response.raise_for_status()
#     departures = response.json().get('departures', [])
    
#     for flight in departures:
#         if flight.get('number') == flight_number:
#             departure_time_str = flight['departure']['scheduledTimeLocal'].split('T')[1].split('+')[0]
#             arrival_time_str = flight['arrival']['scheduledTimeLocal'].split('T')[1].split('+')[0]

#             departure_dt = datetime.strptime(departure_time_str, '%H:%M')
#             arrival_dt = datetime.strptime(arrival_time_str, '%H:%M')

#             # Calculate scheduled elapsed time in minutes
#             duration_minutes = (arrival_dt - departure_dt).total_seconds() / 60
#             if duration_minutes < 0: duration_minutes += 1440 # Handle overnight flights
            
#             return {
#                 'Time': departure_dt.hour * 60 + departure_dt.minute,
#                 'Length': int(duration_minutes)
#             }
#     raise Exception(f"Flight {flight_number} not found for the given date.")


# def get_weather_forecast(iata_code, flight_date_str):
#     """Fetches weather forecast using airport coordinates (a real implementation would need a coordinate lookup)."""
#     # NOTE: This is a simplified version. A production app would first look up IATA -> lat/lon.
#     # For now, we'll use a placeholder. You must replace this with a real lookup.
#     coords = {'JFK': (40.64, -73.78), 'LAX': (33.94, -118.40)}.get(iata_code)
#     if not coords: raise Exception(f"Coordinates for {iata_code} not found.")

#     lat, lon = coords
#     url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
#     response = requests.get(url)
#     response.raise_for_status()
#     forecast_list = response.json()['list']

#     # Find the forecast closest to the flight date (this is a simplified logic)
#     # A real app would parse the flight_date and find the correct forecast entry
#     weather = forecast_list[8] # Default to ~24 hours ahead
    
#     return {
#         'Temp': weather['main']['temp'],
#         'Wind': weather['wind']['speed'],
#         'Condition': weather['weather'][0]['main']
#     }

# # --- 4. Main Quote Endpoint ---
# @app.route('/get-quote', methods=['POST'])
# def get_quote():
#     """
#     Main endpoint to receive basic flight info, enrich it, and return a full quote.
#     """
#     try:
#         # Step 1: Get basic info from frontend
#         data = request.get_json()
#         required_fields = ['Airline', 'AirportFrom', 'AirportTo', 'flight_number', 'date', 'premium']
#         if not all(field in data for field in required_fields):
#             return jsonify({'error': 'Missing required fields in request.'}), 400

#         user_premium = float(data['premium'])
#         flight_date = data['date'] # "YYYY-MM-DD"

#         # Step 2: Enrich data - Get flight Time and Length
#         flight_details = get_flight_details(data['AirportFrom'], flight_date, data['flight_number'])

#         # Step 3: Enrich data - Get weather for origin and destination
#         origin_weather = get_weather_forecast(data['AirportFrom'], flight_date)
#         dest_weather = get_weather_forecast(data['AirportTo'], flight_date)
        
#         # Step 4: Assemble the complete feature set for the model
#         model_input = {
#             'Airline': data['Airline'],
#             'AirportFrom': data['AirportFrom'],
#             'AirportTo': data['AirportTo'],
#             'DayOfWeek': datetime.strptime(flight_date, '%Y-%m-%d').weekday() + 1,
#             'Time': flight_details['Time'],
#             'Length': flight_details['Length'],
#             'WeatherFrom_Temp': origin_weather['Temp'],
#             'WeatherFrom_Wind': origin_weather['Wind'],
#             'WeatherFrom_Condition': origin_weather['Condition'],
#             'WeatherTo_Temp': dest_weather['Temp'],
#             'WeatherTo_Wind': dest_weather['Wind'],
#             'WeatherTo_Condition': dest_weather['Condition']
#         }
        
#         # Step 5: Preprocess, Predict, and Calculate Payout
#         input_df = pd.DataFrame([model_input])
#         processed_input = preprocessor.transform(input_df)
#         risk = model.predict(processed_input)[0][0]
        
#         margin = 0.15
#         payout = (user_premium / risk) * (1 - margin) if risk > 0 else 0
        
#         # Step 6: Return the full quote to the frontend
#         return jsonify({
#             'predicted_risk': float(risk),
#             'premium': user_premium,
#             'payout': payout
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




# # # app.py - V2.2 with Single-Day Search

# # import os
# # import json
# # import requests
# # import numpy as np
# # import pandas as pd
# # import tensorflow as tf
# # import joblib
# # import traceback
# # from flask import Flask, request, Response, jsonify
# # from flask_cors import CORS
# # from dotenv import load_dotenv

# # load_dotenv()
# # AERODATABOX_API_KEY = os.getenv("AERODATABOX_API_KEY")

# # class NumpyEncoder(json.JSONEncoder):
# #     def default(self, obj):
# #         # ... (this class is unchanged)
# #         if isinstance(obj, np.integer): return int(obj)
# #         if isinstance(obj, np.floating): return float(obj)
# #         if isinstance(obj, np.ndarray): return obj.tolist()
# #         return super(NumpyEncoder, self).default(obj)

# # print("Loading model...")
# # model = tf.keras.models.load_model('flight_risk_model.h5')
# # print("Model loaded successfully.")
# # print("Loading preprocessor...")
# # preprocessor = joblib.load('preprocessor.joblib')
# # print("Preprocessor loaded successfully.")

# # app = Flask(__name__)
# # CORS(app, resources={r"/*": {"origins": "*"}})

# # @app.route('/get-flights', methods=['POST'])
# # def get_flights():
# #     # This endpoint is now back to fetching flights for a single date.
# #     if not AERODATABOX_API_KEY:
# #         return jsonify({'error': 'API key for flight data is not configured.'}), 500

# #     try:
# #         search_params = request.get_json()
# #         origin = search_params.get('origin')
# #         destination = search_params.get('destination')
# #         flight_date = search_params.get('date')

# #         print(f"--> Received flight search for: {origin} -> {destination} on {flight_date}")

# #         if not all([origin, destination, flight_date]):
# #             return jsonify({'error': 'Missing origin, destination, or date.'}), 400

# #         url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{origin}/{flight_date}"
# #         querystring = {"withLeg":"true", "direction":"Departure", "withCancelled":"false", "withCodeshared":"true", "withCargo":"false", "withPrivate":"false"}
# #         headers = {
# #             "X-RapidAPI-Key": AERODATABOX_API_KEY,
# #             "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
# #         }

# #         api_response = requests.get(url, headers=headers, params=querystring)
# #         api_response.raise_for_status() 
        
# #         departures = api_response.json().get('departures', [])
# #         available_flights = []
# #         for flight in departures:
# #             if flight.get('arrival', {}).get('airport', {}).get('iata') == destination:
# #                 available_flights.append({
# #                     'airline': flight['airline']['name'],
# #                     'number': flight['number'],
# #                     'departureTime': flight['departure']['scheduledTimeLocal'].split('+')[0],
# #                     'arrivalTime': flight['arrival']['scheduledTimeLocal'].split('+')[0],
# #                     'aircraft': flight.get('aircraft', {}).get('model', 'N/A'),
# #                     'flightDate': flight_date # The date is now the same for all results
# #                 })
        
# #         print(f"<-- Found {len(available_flights)} matching flights.")
# #         return jsonify({'flights': available_flights})

# #     except requests.exceptions.HTTPError as http_err:
# #         if http_err.response.status_code == 404:
# #             print(f"<-- No schedules found for {flight_date}.")
# #             return jsonify({'flights': []})
# #         else:
# #             return jsonify({'error': f'Flight API error: {http_err.response.text}'}), http_err.response.status_code
            
# #     except Exception as e:
# #         return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# # # --- (Your /get-premium endpoint is unchanged) ---
# # @app.route('/get-premium', methods=['POST', 'OPTIONS'])
# # def get_premium():
# #     # ... (This function is unchanged)
# #     if request.method == 'OPTIONS': return '', 204
# #     try:
# #         input_data_json = request.get_json()
# #         user_premium = float(input_data_json.get('premium', 100))
# #         flight_data = pd.DataFrame([input_data_json])
# #         input_data_processed = preprocessor.transform(flight_data)
# #         prediction_probability = model.predict(input_data_processed)[0][0]
# #         margin = 0.15 
# #         payout = 0
# #         if prediction_probability > 0:
# #              payout = (user_premium / prediction_probability) * (1 - margin)
# #         response_body = {'predicted_risk': prediction_probability, 'premium': user_premium, 'payout': payout}
# #         json_response = json.dumps(response_body, cls=NumpyEncoder)
# #         return Response(json_response, mimetype='application/json', status=200)
# #     except Exception as e:
# #         error_details = {'error': str(e), 'traceback': traceback.format_exc()}
# #         return Response(json.dumps(error_details), mimetype='application/json', status=500)

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000)














# # # app.py - FINAL CORRECTED VERSION
# # import json
# # import numpy as np
# # import pandas as pd
# # import tensorflow as tf
# # import joblib
# # import traceback # <--- Add this import for detailed error logging
# # from flask import Flask, request, Response
# # from flask_cors import CORS

# # class NumpyEncoder(json.JSONEncoder):
# #     def default(self, obj):
# #         if isinstance(obj, np.integer): return int(obj)
# #         if isinstance(obj, np.floating): return float(obj)
# #         if isinstance(obj, np.ndarray): return obj.tolist()
# #         return super(NumpyEncoder, self).default(obj)

# # print("Loading model...")
# # model = tf.keras.models.load_model('flight_risk_model.h5')
# # print("Model loaded successfully.")

# # print("Loading preprocessor...")
# # preprocessor = joblib.load('preprocessor.joblib')
# # print("Preprocessor loaded successfully.")

# # app = Flask(__name__)
# # CORS(app, resources={r"/get-premium": {"origins": "*"}})

# # @app.route('/get-premium', methods=['POST', 'OPTIONS'])
# # def get_premium():
# #     if request.method == 'OPTIONS':
# #         return '', 204

# #     try:
# #         input_data_json = request.get_json()
# #         user_premium = float(input_data_json.get('premium', 100))
# #         flight_data = pd.DataFrame([input_data_json])

# #         input_data_processed = preprocessor.transform(flight_data)
# #         prediction_probability = model.predict(input_data_processed)[0][0]
        
# #         margin = 0.15 
# #         payout = 0
# #         if prediction_probability > 0:
# #              payout = (user_premium / prediction_probability) * (1 - margin)
        
# #         response_body = {
# #             'predicted_risk': prediction_probability,
# #             'premium': user_premium,
# #             'payout': payout
# #         }
# #         json_response = json.dumps(response_body, cls=NumpyEncoder)
# #         return Response(json_response, mimetype='application/json', status=200)

# #     except Exception as e:
# #         # Provide a detailed error if something goes wrong on the backend
# #         error_details = {'error': str(e), 'traceback': traceback.format_exc()}
# #         return Response(json.dumps(error_details), mimetype='application/json', status=500)

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000)
