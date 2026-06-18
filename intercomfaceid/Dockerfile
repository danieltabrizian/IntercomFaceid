# Pinned to bookworm: its Intel NEO OpenCL driver still supports Gemini Lake
# (Gen9.5). trixie ships a newer NEO that dropped legacy-platform support, and
# its 'main' component no longer carries intel-opencl-icd.
FROM python:3.9-slim-bookworm

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

# Intel iGPU OpenCL runtime (NEO) for the experimental OpenVINO GPU benchmark.
# intel-opencl-icd lives in non-free; enable it by adding the components to the
# existing deb822 sources (a separate .list conflicts on Signed-By).
RUN sed -i 's/^Components: .*/Components: main contrib non-free non-free-firmware/' \
      /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y intel-opencl-icd ocl-icd-libopencl1 clinfo && \
    apt-get clean

RUN pip install --upgrade pip

RUN pip install --no-cache-dir insightface onnxruntime opencv-python paho-mqtt pyserial fastapi "uvicorn[standard]"

# OpenVINO runtime for the experimental iGPU benchmark (standalone — does NOT
# replace onnxruntime, so production CPU recognition is unaffected).
RUN pip install --no-cache-dir "openvino==2024.6.0"

WORKDIR /usr/src/app
COPY . .

CMD ["python", "-u", "main.py"]
