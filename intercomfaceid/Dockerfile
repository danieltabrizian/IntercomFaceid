FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies (DeepFace, OpenCV, paho-mqtt, requests)
RUN pip install deepface opencv-python paho-mqtt requests

# Run the Python script
# Run the Python script with unbuffered output (-u)
CMD ["python", "-u", "./main.py"]