import random
import cv2
import numpy as np
import torch
from PIL import Image


def crop_4_patches(image):
    crop_size = int(image.size[0]/2)
    return (image.crop((0, 0, crop_size, crop_size)), image.crop((0, crop_size, crop_size, 2*crop_size)),
            image.crop((crop_size, 0, 2*crop_size, crop_size)), image.crop((crop_size, crop_size, 2*crop_size, 2*crop_size)))


def pre_processing(image, transform):
    high_level = []
    middle_level = []
    low_level = []
    crops_4 = crop_4_patches(image)
    for c_4 in crops_4:
        crops_8 = crop_4_patches(c_4)
        high_level.append(transform(crops_8[0]))
        high_level.append(transform(crops_8[3]))
        for c_8 in [crops_8[1], crops_8[2]]:
            crops_16 = crop_4_patches(c_8)
            middle_level.append(transform(crops_16[0]))
            middle_level.append(transform(crops_16[3]))
            for c_16 in [crops_16[1], crops_16[2]]:
                crops_32 = crop_4_patches(c_16)
                low_level.append(transform(crops_32[0]))
                low_level.append(transform(crops_32[3]))
    return torch.stack(high_level), torch.stack(middle_level), torch.stack(low_level)
