FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

VOLUME [ "/1mdfFinal/dataset", "/1mdfFinal/visual/models_luka_1mdf" ]


#get newest version of ffmpeg
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

# Copy the environment file to the Docker image
COPY . /1mdfFinal/

# Set the working directory
WORKDIR /1mdfFinal


RUN pip config set global.timeout 120
RUN pip config set global.retries 10


# Create and activate the Conda environment
RUN conda env create -f /1mdfFinal/environment.yml

# Make RUN commands use the new environment:
    # source /opt/conda/bin/activate pytorch_env_1mdf
SHELL ["conda", "run", "-n", "pytorch_env_1mdf", "/bin/bash", "-c"]
# Ensure the environment is activated when the container starts
CMD ["bash"]
