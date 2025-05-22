# Use the latest official PyTorch image with CUDA (as of 25.03, Ubuntu 24.04, Python 3.12, CUDA 12.8)
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Download and compile FFmpeg 7.1.1 from source
# Install basic build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    xz-utils \
    nasm \
    yasm \
    python3-dev \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Build FFmpeg from source (version 7.1 - latest stable with full PCM support)
RUN cd /tmp && \
    wget https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz && \
    tar xf ffmpeg-7.1.tar.xz && \
    cd ffmpeg-7.1 && \
    ./configure \
        --prefix=/usr/local \
        --enable-shared \
        --enable-gpl \
        --enable-version3 \
        --enable-nonfree \
        --disable-static \
        --disable-debug \
        --disable-doc \
        --disable-programs && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/ffmpeg-7.1*

# Download and install the latest static ffmpeg for runtime
RUN cd /tmp && \
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    cp ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-*-static/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg-* && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# Verify FFmpeg development libraries are properly installed
RUN pkg-config --exists libavcodec libavformat libavutil && \
    echo "✓ FFmpeg development libraries found" || \
    (echo "✗ FFmpeg development libraries missing" && exit 1)

# Show what FFmpeg version we're building against (should be 7.1)
RUN pkg-config --modversion libavcodec libavformat libavutil | head -3

# Install uv and set up virtual environment
RUN pip install --upgrade pip && pip install uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-home
RUN uv venv /opt/venv --python 3.12
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone and build Decord with FFmpeg 7.x fixes
RUN git clone https://github.com/TheLukaDragar/decord.git /tmp/decord && \
    cd /tmp/decord && \
    git submodule init && \
    git submodule update --recursive && \
    mkdir -p build && \
    cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_ROOT=/usr/local && \
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
