import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model, Sequential
import utils
import data_config

class InvasiveModel(object):
    baseFolder = 'model/'
    savedModel = 'invasives.sm'
    submission = 'submission.csv'


    def __init__(self):
        self._pretrainModel = None
        self._trainModel = None
        self._predictModel = None

    def addLocal(self, model, next_shape, dropout):
        model.add(Dense(next_shape, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    def saveSubmission(self, outputs):
        outputs.sort(key=lambda x: x[0])
        np.savetxt(self.baseFolder + self.submission, outputs, 
            fmt ='%d,%.4f', header='name,invasive', comments='')

    def save(self):
        path = self.baseFolder + self.savedModel
        self._trainModel.save_weights(path)
        return path

    def getPretrainDataPerFolder(self, data):
        x = utils.load_array(self.baseFolder + data.pretrain[0])
        y = utils.load_array(self.baseFolder + data.pretrain[1])
        f = utils.load_array(self.baseFolder + data.pretrain[2])
        return [x, y, f]





 

