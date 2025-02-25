# Use the official NVIDIA CUDA runtime image as the base image, including CUDA 11.8 and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Update package lists and install Python 3.8 and necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Ensure pip is up to date
RUN python3 -m pip install --upgrade pip

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Copy all project files into the container
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Define the default command to run when the container starts
CMD ["python3", "app/cli_chat.py"]