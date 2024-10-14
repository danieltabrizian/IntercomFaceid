FROM python:3.9-slim

# Install system dependencies (build tools, g++, OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean

# Install InsightFace and other Python dependencies
RUN pip install --no-cache-dir insightface onnxruntime opencv-python paho-mqtt

# Set the working directory
WORKDIR /usr/src/app

# Copy the project files into the container
COPY . .

# Run the Python script
CMD ["python", "main.py"]