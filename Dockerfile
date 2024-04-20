FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

# Fix basicsr torchvision import error
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /opt/conda/lib/python3.10/site-packages/basicsr/data/degradations.py
# Copy the project files
COPY ./RealESRGAN_x4plus.pth /app/RealESRGAN_x4plus.pth
COPY ./main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
