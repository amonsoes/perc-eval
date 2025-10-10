import argparse
import os
import torch
import torchvision.transforms as T

from torchvision.io import read_image
from torchvision.utils import save_image


def resize_images(root, size=224):
    resize = T.CenterCrop(size)
    convert_float = T.ConvertImageDtype(torch.float32)
    for ig_name in os.listdir(root):
        if ig_name.endswith('.png'):
            ig_path = root + ig_name
            tensor = convert_float(read_image(ig_path))
            resized_tensor = resize(tensor)
            save_image(resized_tensor, ig_path)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./images/')
    args = parser.parse_args()

    resize_images(root=args.path)

