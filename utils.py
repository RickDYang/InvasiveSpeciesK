import numpy as np
import contextlib
import shutil
import os
import bcolz
import keras
import keras.preprocessing.image as image
#from multiprocessing import Pool
#from multiprocessing.dummy import Pool as ThreadPool

from keras.applications.vgg16 import preprocess_input

invasive_size = (866, 1154)

def getBatches(path, target_size, shuffle = False, batch_size = 1):
    gen = image.ImageDataGenerator()
    return gen.flow_from_directory(path, target_size= target_size,
        class_mode = 'categorical', shuffle = shuffle, batch_size = batch_size)

def get_distorb_batches(path, batch_size, target_size, shuffle = True):
    gen = image.ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1,
        height_shift_range = 0.1, horizontal_flip = True)
    return gen.flow_from_directory(path, target_size= target_size,
        class_mode = 'categorical', shuffle = shuffle, batch_size = batch_size)

def save_array(f, arr):
    try_mkdir(os.path.dirname(f))
    c = bcolz.carray(arr, rootdir=f, mode='w')
    c.flush()

@contextlib.contextmanager
def pretrainWriter(base_folder, folders):
    try_mkdir(base_folder)
    writers = [None for _ in folders]

    def append(data):
        for i, d in enumerate(data):
            if writers[i] is None:
                writers[i] = bcolz.carray(d, mode='w',
                    rootdir= base_folder + folders[i])
            else:
                writers[i].append(d)
            writers[i].flush()

    yield append

    for f in writers:
        if f is not None:
            f.flush()

def load_array(f):
    return bcolz.open(f)

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