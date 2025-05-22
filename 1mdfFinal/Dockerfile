# Use the latest official PyTorch image with CUDA (as of 25.03, Ubuntu 24.04, Python 3.12, CUDA 12.8)
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists
RUN apt-get update

# Remove any existing ffmpeg
RUN apt-get remove -y ffmpeg libavcodec* libavformat* libavutil* || true

# Install basic build dependencies
RUN apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    python3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Create directory for FFmpeg packages
RUN mkdir -p /tmp/ffmpeg_packages

# Download and install FFmpeg 7.1.1 prebuilt packages from Ubuntu 24.04
RUN cd /tmp/ffmpeg_packages && \
    # Download the main FFmpeg 7.1.1 library packages
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavutil59_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavcodec61_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavformat61_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavfilter10_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libswresample5_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libswscale8_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libpostproc58_7.1.1-1ubuntu1_amd64.deb && \
    # Download development packages
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavutil-dev_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavcodec-dev_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavformat-dev_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libavfilter-dev_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libswresample-dev_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libswscale-dev_7.1.1-1ubuntu1_amd64.deb && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/f/ffmpeg/libpostproc-dev_7.1.1-1ubuntu1_amd64.deb && \
    # Install all packages
    dpkg -i *.deb || apt-get install -f -y && \
    # Clean up
    cd / && \
    rm -rf /tmp/ffmpeg_packages

# Update library cache
RUN ldconfig

# Set PKG_CONFIG_PATH environment variable
ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig

# Verify FFmpeg installation and version
RUN pkg-config --modversion libavcodec libavformat libavutil

# Download and install the latest static ffmpeg
RUN apt-get update && \
    apt-get install -y wget xz-utils && \
    cd /tmp && \
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    cp ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-*-static/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg-* && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Verify ffmpeg version
RUN ffmpeg -version

# Install uv and set up virtual environment
RUN pip install --upgrade pip && pip install uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-home
RUN uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone and build Decord with FFmpeg 7.x fixes
RUN git clone https://github.com/TheLukaDragar/decord.git /tmp/decord && \
    cd /tmp/decord && \
    mkdir -p build && \
    cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/usr && \
    make -j$(nproc) && \
    cd ../python && \
    uv pip install .

# Copy code and requirements
COPY . /1mdfFinal/
WORKDIR /1mdfFinal

# Install remaining Python dependencies with uv
RUN uv pip install -r requirements.txt

# Entrypoint
CMD ["bash"]
