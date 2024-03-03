# RealESRGAN image upscale with Docker

RealESRGAN is a deep-learning model for image upscaling. This repository contains a simple Python script to upscale images using RealESRGAN, wrapped in a Docker container.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- Nvidia GPU (optional)

## Build

Download the RealESRGAN model and build the container:

```sh
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
docker build -t upscaler .
```

## Usage

```sh
docker run --rm -v ./images:/images upscaler --scale 4
```

Replace `./images` with the path to the directory containing the images you want to upscale.

Note that the folder should only contain images and no other files.

The upscaled images will be saved in the same directory as the original images, with the prefix `upscaled_`.

If you do not have an Nvidia GPU, you can use the `--cpu` flag to run the container on the CPU.
