import os
import numpy as np
import PIL
from PIL import Image
from data_config import *
import utils

def divid_images():
    labels = np.genfromtxt(label_path, dtype = np.int,
        delimiter=',', skip_header = 1)
    labels = dict(labels)

    divid_images_in_folder(trainData.dataFolder, labels)

    divid_images_in_folder(validationData.dataFolder, labels)

def norm_image(self, img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm


def divid_images_in_folder(folder, labels):
    utils.try_mkdir(os.path.join(folder, true_folder))
    utils.try_mkdir(os.path.join(folder, false_folder))

    for f in os.listdir(folder):
        if f.endswith(image_postfix):
            fid = int(os.path.splitext(f)[0])
            if labels[fid] == 1:
                os.rename(os.path.join(folder, f), os.path.join(folder, true_folder,f))
            else:
                os.rename(os.path.join(folder, f), os.path.join(folder, false_folder,f))



if __name__ == '__main__':
    divid_images()
