import numpy as np
import time
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model, Sequential
import utils
import data_config
from Models.PretrainPatchesModel import *

class InvasiveCombineModel(PretrainPatchesModel):
    '''
    Reuse Imagenet models to train an image with different patches divided
    '''

    def __init__(self, subModels):
        PretrainPatchesModel.__init__(self)
        self.subModels = subModels

    def getTrainModel(self):

        if self._trainModel == None:
            model = Sequential()
            model.add(Flatten(input_shape=(len(self.subModels), 2, )))
            model.add(Dense(2, activation ='softmax'))

            self._trainModel = model

        return self._trainModel

    def getPretrainData(self):
        train = self.getPretrainDataPerFolder(data_config.trainData)
        train = train[:-1] #self.randomExpendPretrain(train, 10)
        validation = self.getPretrainDataPerFolder(data_config.validationData)
        return train, validation[:-1]

    def getPretrainDataPerFolder(self, data):
        x = None
        y = None
        f = None
        for model in self.subModels:
            outputs = model.predict(data)
            outputs.sort(key=lambda x: x[0])

            tx = np.asarray([[1-o[1], o[1]] for o in outputs])
            if x is None:
                x = tx
            else:
                t =  [ np.column_stack((x[i], tx[i])) for i in range(len(outputs))]
                x = np.asarray(t)

        f = [o[0] for o in outputs]
        y = [o[2] for o in outputs]

        #print(x.shape)
        #print(np.asarray(y).shape)
        #print(np.asarray(f).shape)
        return [x, np.asarray(y), f]
   