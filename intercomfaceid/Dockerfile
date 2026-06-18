FROM python:3.9-slim

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
    libgl1 \
    libglib2.0-0 \
    wget \
    && apt-get clean

RUN pip install --upgrade pip

RUN pip install --no-cache-dir insightface onnxruntime opencv-python paho-mqtt pyserial fastapi "uvicorn[standard]"

# Download SFace models (YuNet detector + SFace recognizer)
RUN mkdir -p /models && \
    wget -q -O /models/yunet.onnx \
      "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" && \
    wget -q -O /models/sface.onnx \
      "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

WORKDIR /usr/src/app
COPY . .

CMD ["python", "-u", "main.py"]
