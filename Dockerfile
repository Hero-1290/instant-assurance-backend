# Dockerfile

# 1. Use an official, lightweight Python runtime as a base
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies first
# This is an efficient practice called "layer caching"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Also install Gunicorn, a production-grade server for Flask
RUN pip install gunicorn

# 4. Copy all the other files from your project into the container
# This includes app.py, the .h5 model, and the .joblib preprocessor
COPY . .

# 5. Tell the container what command to run when it starts
# It will run your Flask app using Gunicorn on the port provided by Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

