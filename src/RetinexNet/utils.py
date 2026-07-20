import numpy as np
from PIL import Image

def data_augmentation(image, mode):
    # image can be numpy array (H, W, C)
    if mode == 0:
        return image
    elif mode == 1:
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))
    return image

def load_images(file):
    im = Image.open(file).convert('RGB')
    return np.array(im, dtype="float32") / 255.0

def save_images(filepath, result_1, result_2 = None):
    # expect result_1 and result_2 to be numpy arrays (H, W, C) in range [0, 1]
    result_1 = np.squeeze(result_1)
    if result_2 is not None:
        result_2 = np.squeeze(result_2)

    if result_2 is None:
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')
