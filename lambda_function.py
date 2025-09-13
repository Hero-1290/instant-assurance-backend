import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- GLOBAL SCOPE ---
# Load the trained model. This happens only once when the Lambda is "cold started".
model = tf.keras.models.load_model('flight_risk_model.h5')

# We need to recreate the same preprocessor we used in training.
# IMPORTANT: This must match the columns and transformations from your Colab notebook.
categorical_features = ['Airline', 'AirportFrom', 'AirportTo']
numerical_features = ['DayOfWeek', 'Time', 'Length']

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Note: We need a dummy DataFrame to 'fit' the preprocessor.
# This is a one-time setup to make sure the preprocessor knows all the categories.
# In a production system, you would save and load the fitted preprocessor object.
dummy_data = {
    'Airline': ['DL', 'AA'], 'AirportFrom': ['ATL', 'JFK'], 'AirportTo': ['LAX', 'SFO'],
    'DayOfWeek': [1, 2], 'Time': [1200, 1400], 'Length': [150, 160]
}
dummy_df = pd.DataFrame(dummy_data)
preprocessor.fit(dummy_df)
# --- END GLOBAL SCOPE ---


def lambda_handler(event, context):
    """
    This function is the main entry point for our Lambda.
    It receives flight data from the API Gateway.
    """
    try:
        # 1. Parse the input data from the API request body
        body = json.loads(event['body'])
        input_data = pd.DataFrame([body]) # Convert the single request into a DataFrame

        # 2. Preprocess the input data using our pre-fitted preprocessor
        input_data_processed = preprocessor.transform(input_data)

        # 3. Make a prediction using the loaded DL model
        prediction_probability = model.predict(input_data_processed)[0][0]

        # 4. Calculate the premium using our business logic
        payout_amount = 1000  # Fixed payout
        margin = 1.3          # 30% margin
        premium = (prediction_probability * payout_amount) * margin

        # 5. Return a successful JSON response
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*', # Enable CORS
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'predicted_risk': float(prediction_probability),
                'premium': round(premium, 2),
                'payout': payout_amount
            })
        }

    except Exception as e:
        # Return an error response if something goes wrong
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({'error': str(e)})
        }