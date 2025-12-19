FROM --platform=linux/amd64 python:3.11-slim

# 1. Set the working directory
WORKDIR /app

# 2. Copy requirements FIRST (to take advantage of caching)
COPY requirements.txt .

# 3. Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code
COPY . .

# 5. Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

