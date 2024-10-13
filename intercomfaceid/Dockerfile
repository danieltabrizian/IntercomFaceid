FROM python:3.11-alpine

# Install system dependencies for dlib and OpenCV
RUN apk add --no-cache \
    build-base \
    cmake \
    linux-headers \
    libjpeg-turbo-dev \
    openblas-dev \
    lapack-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr-dev \
    libxi-dev \
    gtk+3.0-dev \
    python3-dev \
    boost-dev \
    && apk add jpeg-dev zlib-dev

RUN pip install --upgrade pip setuptools wheel
# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Show installed requests package (for debug)
RUN pip show requests

# Run the Python script
CMD ["python", "./main.py"]