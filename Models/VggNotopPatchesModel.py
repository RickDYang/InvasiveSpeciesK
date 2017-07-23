import numpy as np
import time
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.models import Model, Sequential
import utils
import data_config
from Models.InvasiveModels import *
from Models.VggFullPatchesModel import *

class VggNotopPatchesModel(VggFullPatchesModel):
    '''
    Reuse Imagenet Vgg models with top to train an image with different patches divided
    Kaggle score: (70, 7, 7, 512, )  0.98215 - 256(0.7, norm)->flatten->256(0.7, norm)->2
    ''',
    baseFolder = 'model/vggnotoppatches/'
    ratio = 0.25

    def __init__(self):
        PretrainPatchesModel.__init__(self)

    def getTrainModel(self):

        if self._trainModel == None:
            model = Sequential()
            model.add(Reshape((234, 7 * 7 * 512), input_shape=(234, 7, 7, 512, )))

            self.addLocal(model, 64, 0.8)
            model.add(Flatten())
            self.addLocal(model, 32, 0.8)

            model.add(Dense(2, activation ='softmax'))

            self._trainModel = model

        return self._trainModel

    def getPretrainModel(self):
        if self._pretrainModel == None: 
            self._pretrainModel = VGG16(weights='imagenet', include_top=False)
        return self._pretrainModel