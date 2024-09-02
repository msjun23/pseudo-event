import numpy as np
from PIL import Image


def save_as_rb_img(data, name):
    # [2, H, W]
    data = data.detach().cpu()
    C, H, W = data.shape
    
    # 3channel (RGB) image (H, W, 3)
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[data[0] != 0, 0] = 255  # red channel
    image_array[data[1] != 0, 2] = 255  # blue channel
    
    # numpy to PIL
    image = Image.fromarray(image_array)
    image.save(name)


def return_as_rb_img(data):
    # [2, H, W]
    data = data.detach().cpu()
    C, H, W = data.shape
    
    # 3channel (RGB) image (H, W, 3)
    image_array = np.zeros((H, W, 3), dtype=np.uint8)
    image_array[data[0] != 0, 0] = 255  # red channel
    image_array[data[1] != 0, 2] = 255  # blue channel
    
    return image_array