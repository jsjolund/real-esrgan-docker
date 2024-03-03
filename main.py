import argparse
import os

import numpy
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer

parser = argparse.ArgumentParser(description="Upscale images using RealESRGAN")
parser.add_argument(
    "-s",
    "--scale",
    type=int,
    default=4,
    required=False,
    help="Image upscale factor (2 or 4)",
)
parser.add_argument(
    "-c",
    "--cpu",
    type=bool,
    default=False,
    required=False,
    help="Use CPU instead of GPU",
)
args = parser.parse_args()
assert args.scale in [2, 4], "Upscale factor must be 2 or 4"

upscale_model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    scale=4,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
)
upsampler = RealESRGANer(
    scale=4,
    model_path="/app/RealESRGAN_x4plus.pth",
    dni_weight=None,
    model=upscale_model,
    tile=0,
    tile_pad=10,
    pre_pad=10,
    half=True,
    gpu_id=0 if not args.cpu else None,
)


def upscale_image(image_path: str, outscale: int = 4) -> Image:
    image = Image.open(image_path)
    upscaled_arr = upsampler.enhance(numpy.array(image), outscale=outscale)[0]
    return Image.fromarray(upscaled_arr)


print(f"Upscaling images by {args.scale}")
for image in os.listdir("/images"):
    upscaled_image = upscale_image(f"/images/{image}", outscale=args.scale)
    upscaled_image.save(f"/images/upscaled_{image}")
    print(f"Upscaled {image} -> upscaled_{image}")
