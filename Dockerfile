# Use a specific Python version with platform specification
FROM python:3.10.16-slim

# Install system dependencies and clean up to reduce image size
RUN apt-get update && \
    apt-get -qq -y install tesseract-ocr libtesseract-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application code
COPY . /app

# Copy and install Python dependencies
RUN pip install -r requirements.txt


# Start Gunicorn to serve the app
CMD ["gunicorn", "main:app"]