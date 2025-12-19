# Change your first line to this:
FROM --platform=linux/amd64 python:3.11-slim

# ... the rest of your Dockerfile stays the same

# Use the official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system tools (needed for some Python libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy EVERYTHING from your GitHub repo into the container
# (This includes main.py, requirements.txt, etc.)
COPY . .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Tell Google Cloud to listen on the correct port
ENV PORT=8080

# Start the application
# Ensure your python file is named 'main.py' and the app instance is 'app'
# Use this exact line (no brackets, no 'exec' keyword)
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

