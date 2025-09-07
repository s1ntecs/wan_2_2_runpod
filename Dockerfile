FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /
RUN apt-get update && \
    apt upgrade -y && \
    apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install torch==2.4.0 \
    torchvision \
    git+https://github.com/huggingface/diffusers.git@refs/pull/11059/head \
    transformers \
    tokenizers>=0.20.3 \ 
    accelerate==1.4.0 \
    huggingface-hub==0.29.1 \
    requests==2.32.3 \
    "numpy<2" \
    ftfy \
    peft \
    regex \
    "pillow>=10.0.0" \
    "imageio>=2.31.1" \
    "imageio-ffmpeg>=0.4.8" \
    opencv-python>=4.9.0.80 \
    tqdm \
    easydict \
    gradio>=5.0.0 \
    runpod \
    xformers \
    dashscope \
    flash_attn \
    numpy

RUN pip install "huggingface_hub[cli]"

# Скачиваем pget (аналог run: curl ... из cog)
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && chmod +x /usr/local/bin/pget


# Create model directory

RUN if [ ! -d "loras" ]; then mkdir loras; fi

RUN python3 download_checkpoints.py

COPY --chmod=755 start_standalone.sh /start.sh

# Start the container
ENTRYPOINT /start.sh