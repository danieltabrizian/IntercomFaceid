# Pinned to bookworm for a stable, reproducible base (avoids surprises when the
# python:3.9-slim tag rolls forward to a newer Debian).
FROM python:3.9-slim-bookworm

# System dependencies (build tools, g++, OpenCV runtime libs).
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
    && apt-get clean

RUN pip install --upgrade pip

RUN pip install --no-cache-dir insightface onnxruntime opencv-python paho-mqtt pyserial fastapi "uvicorn[standard]"

# ffmpeg + go2rtc for the bundled camera restreamer (USB -> RTSP/WebRTC/HLS/MJPEG).
RUN apt-get update && apt-get install -y ffmpeg wget && apt-get clean
RUN wget -q -O /usr/local/bin/go2rtc \
      https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_amd64 \
    && chmod +x /usr/local/bin/go2rtc \
    && /usr/local/bin/go2rtc --version

WORKDIR /usr/src/app
COPY . .

CMD ["python", "-u", "main.py"]
