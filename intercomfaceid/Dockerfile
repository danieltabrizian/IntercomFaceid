FROM python:3.11-slim

# Install system dependencies for dlib, OpenCV, and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean

# Install dlib and OpenCV via pip using precompiled wheels
RUN pip install --no-cache-dir dlib opencv-python face-recognition paho-mqtt requests

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# Show installed requests package (for debug)
RUN pip show requests

# Run the Python script
CMD ["python", "./main.py"]