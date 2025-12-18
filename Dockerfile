# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# We include 'curl' for health checks and 'libmagic' if you handle files
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container image.
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Run the web service on container startup.
# We use uvicorn to run the FastAPI app. 
# Cloud Run tells us which PORT to use via an environment variable.
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
