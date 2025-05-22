# Use the latest official PyTorch image with CUDA (as of 25.03, Ubuntu 24.04, Python 3.12, CUDA 12.8)
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get remove -y ffmpeg || true

# Download and install the latest static ffmpeg
RUN apt-get update && \
    apt-get install -y wget xz-utils && \
    cd /tmp && \
    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    cp ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-*-static/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg-* && \
    apt-get remove -y wget xz-utils && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# (Optional) Verify ffmpeg version
RUN ffmpeg -version

# Install uv (modern Python package manager)
RUN pip install --upgrade pip && pip install uv

# Copy code and requirements
COPY . /1mdfFinal/
WORKDIR /1mdfFinal

RUN uv venv .venv --python 3.12

ENV VIRTUAL_ENV=/1mdfFinal/.venv
ENV PATH="/1mdfFinal/.venv/bin:$PATH"

# Install Python dependencies with uv
RUN uv pip install -r requirements.txt

# Entrypoint
CMD ["bash"]
