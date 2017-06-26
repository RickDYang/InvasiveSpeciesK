import numpy as np
import shutil
import os
import bcolz
import keras
import keras.preprocessing.image as image
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from keras.applications.vgg16 import preprocess_input

vgg_image_size = (224, 224)
invasive_image_size = (866, 1154)
gap = (112, 112)

patch_range = [ list(range(invasive_image_size[0] - gap[0]))[::gap[0]],
                list(range(invasive_image_size[1] - gap[1]))[::gap[1]]]
patch_range[0][-1] = invasive_image_size[0] - vgg_image_size[0]
patch_range[1][-1] = invasive_image_size[1] - vgg_image_size[1]

def get_distorb_batches(path, batch_size, shuffle = True):
    gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.1,
        zoom_range=0.1, channel_shift_range=10, horizontal_flip=True)
    return gen.flow_from_directory(path, target_size= vgg_image_size,
        class_mode = 'categorical', shuffle = shuffle, batch_size = batch_size)

def get_data(batches):
    x = np.concatenate([batches.next()[0] for i in range(batches.samples)])
    y = np.concatenate([batches.next()[1] for i in range(batches.samples)])
    return x, y

def load_image_for_vgg(path):
    img = image.load_img(path, target_size=vgg_image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x

def save_array(f, arr):
    try_mkdir(os.path.dirname(f))
    c = bcolz.carray(arr, rootdir=f, mode='w')
    c.flush()

def load_array(f):
    return bcolz.open(f)[:]

def try_mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def clear_or_mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.mkdir(folder)
    else:
        os.mkdir(folder)

def copyfile(scr, dst):
    try_mkdir(os.path.dirname(dst))
    shutil.copyfile(scr, dst)


def getFileId(f):
    return int(os.path.splitext(os.path.basename(f))[0])

img_dx = list(range(invasive_image_size[0] - gap[0]))[::gap[0]]
img_dx[-1] = invasive_image_size[0] - vgg_image_size[0]
img_dy = list(range(invasive_image_size[1] - gap[1]))[::gap[1]]
img_dy[-1] = invasive_image_size[1] - vgg_image_size[1]

def get_patch(img):
    def _get(x):
        return np.concatenate([img[:, x:x+vgg_image_size[0], y:y+vgg_image_size[0], ::]
            for y in img_dy])

    return _get

def getBatches(path, target_size = vgg_image_size, shuffle = False, batch_size = 1):
    gen = image.ImageDataGenerator()
    return gen.flow_from_directory(path, target_size= target_size,
        class_mode = 'categorical', shuffle = shuffle, batch_size = batch_size)


def get_images_patches(path):
    gen = image.ImageDataGenerator()
    batches = gen.flow_from_directory(path, target_size= invasive_image_size,
        class_mode = 'categorical', shuffle = False, batch_size = 1)
    n = batches.samples
    for _ in range(n):
        full, labels = batches.next()
        img = np.concatenate([full[:, x:x+vgg_image_size[0], y:y+vgg_image_size[1], :]
            for x in img_dx for y in img_dy])

        yield img, labels, n