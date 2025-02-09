# Use a specific Python version with platform specification
FROM --platform=linux/amd64 python:3.10.15

# Install system dependencies and clean up to reduce image size
RUN apt-get update && \
    apt-get -qq -y install tesseract-ocr libtesseract-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Start Gunicorn to serve the app
CMD ["gunicorn", "main:app"]