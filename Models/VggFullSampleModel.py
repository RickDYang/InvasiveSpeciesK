import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential
import utils
import data_config
from Models.InvasiveModels import *
from Models.VggFullPatchesModel import *

class VggFullSampleModel(PretrainPatchesModel):
    '''
    Reuse Imagenet Vgg 19 models to train an image by a single sampling
    refractor but no time test
    Kaggel Score: 0.89229
    '''
    baseFolder = 'model/vggfullsample/'
    image_shape = (224, 224)

    def __init__(self):
        PretrainPatchesModel.__init__(self)

    def getTrainModel(self):
        if self._trainModel == None:
            model = Sequential()
            model.add(Dense(256, activation ='relu', input_shape=(1000, )))
            model.add(Dropout(0.5))
            model.add(Dense(2, activation ='softmax'))

            self._trainModel = model

        return self._trainModel

    def getPretrainModel(self):
        if self._pretrainModel == None: 
            self._pretrainModel = VGG19(weights='imagenet', include_top=True)
        return self._pretrainModel

    def processPretrain(self, data, features, labels, fileIds):
        utils.save_array(self.baseFolder + data.pretrain[0], np.concatenate(features))
        utils.save_array(self.baseFolder + data.pretrain[1], np.concatenate(labels))
        utils.save_array(self.baseFolder + data.pretrain[2], fileIds)


    def getImagesPatches(self, path):
        batches = utils.getBatches(path, self.image_shape)
        for i in range(batches.samples):
            images, labels = batches.next()
            yield images, labels, batches.filenames[i]

