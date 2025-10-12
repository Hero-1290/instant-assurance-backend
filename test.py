# test_api.py
import requests
import os

# --- IMPORTANT: PASTE YOUR API KEY HERE ---
API_KEY = "80847befcfmsh6a5df50195aaf00p1c1bd9jsn384a8c1d8a99" 

# --- Test Data ---
origin = "JFK"
flight_date = "2025-10-11" # Yesterday's date

url = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/{origin}/{flight_date}"
params = {"withLeg": "true", "direction": "Departure"}
headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
}

print(f"▶️  Testing API call for {origin} on {flight_date}...")

try:
    response = requests.get(url, headers=headers, params=params)
    # This will raise an error if the status code is 4xx or 5xx
    response.raise_for_status() 
    
    data = response.json()
    print("✅ SUCCESS! API call worked.")
    print(f"Found {len(data.get('departures', []))} flights.")

except requests.exceptions.HTTPError as e:
    print("\n❌ FAILURE! The API request failed.")
    print(f"   Status Code: {e.response.status_code}")
    print(f"   Reason: {e.response.reason}")
    print(f"   Response Body: {e.response.text}")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")