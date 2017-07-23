import numpy as np
import time
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model, Sequential
import utils
import data_config
from Models.InvasiveModels import *
from Models.PretrainPatchesModel import *

class VggFullPatchesModel(PretrainPatchesModel):
    '''
    Reuse Imagenet Vgg models to train an image with different patches divided
    Kaggle score: 0.94170
    '''
    baseFolder = 'model/vggfullpatches/'

    def __init__(self):
        PretrainPatchesModel.__init__(self)

    def getTrainModel(self):

        if self._trainModel == None:
            model = Sequential()
            model.add(Flatten(input_shape=(70, 1000, )))
            
            self.addLocal(model, 1000, 0.5)
            self.addLocal(model, 200, 0.5)
            
            model.add(Dense(2, activation ='softmax'))

            self._trainModel = model

        return self._trainModel

    def getPretrainModel(self):
        if self._pretrainModel == None: 
            self._pretrainModel = VGG16(weights='imagenet', include_top=True)
            self._pretrainModel.summary()
        return self._pretrainModel
    
    @property
    def getPatchesSize(self):
        return (224, 224)