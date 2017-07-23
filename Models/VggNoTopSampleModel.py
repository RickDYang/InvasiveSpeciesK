import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model, Sequential
import utils
import data_config
from Models.InvasiveModels import *
from Models.VggFullSampleModel import *

class VggNoTopSampleModel(VggFullSampleModel):
    '''
    Reuse Imagenet Vgg 16 models to train an image 
    by a single sampling without top dense layers
    Kaggel Score: 93%
    '''
    baseFolder = 'model/vggnotopsample/'

    def __init__(self):
        VggFullSampleModel.__init__(self)

    def getTrainModel(self):
        if self._trainModel == None:
            model = Sequential()
            model.add(Flatten(input_shape=(7, 7, 512, )))
            model.add(Dense(256, activation ='relu'))
            model.add(Dense(2, activation ='softmax'))

            self._trainModel = model

        return self._trainModel

    def getPretrainModel(self):
        if self._pretrainModel == None: 
            self._pretrainModel = VGG16(weights='imagenet', include_top=False)
        return self._pretrainModel

    def pretrain(self):
        self.pretrainPerFolder(data_config.validationData)
        self.pretrainPerFolder(data_config.testData)
        self.pretrainDistorbData(data_config.trainData, 32)

    def pretrainDistorbData(self, data, batch_size):
        batches = utils.get_distorb_batches(data.dataFolder, batch_size = batch_size, shuffle = False)
        features = list()
        labels = list()
        fileIds = list()

        pretrainModel = self.getPretrainModel()
        for i in range(batches.samples):
            img, label = batches.next()
            s = time.time()
            feature = pretrainModel.predict(img, batch_size = batch_size)
            features.append(feature)
            labels.append(label)
            fileIds.append(utils.getFileId(batches.filenames[i]))
            print('processed images ETA {:.02f}s - {}'.format(time.time() - s, i), end='\r')

        self.processPretrain(data, features, labels, fileIds)


