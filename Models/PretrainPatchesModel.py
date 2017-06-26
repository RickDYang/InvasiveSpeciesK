import numpy as np
import time
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Model, Sequential
import utils
import data_config
from Models.InvasiveModels import *

class PretrainPatchesModel(InvasiveModel):
    '''
    Reuse Imagenet models to train an image with different patches divided
    '''

    def __init__(self):
        InvasiveModel.__init__(self)

    def getTrainModel(self):
        return self._trainModel

    def getPretrainModel(self):
        return self._pretrainModel

    def predict(self, modelPath = None):
        trainModel = self.getTrainModel()
        if modelPath == None:
            modelPath = self.savedModel
        trainModel.load_weights(self.baseFolder + modelPath)

        test, _, f = self.getPretrainDataPerFolder(data_config.testData)
        if test.shape[0] == 0: #has pretrained test data
            return self.directPredict(trainModel)
        else:
            return self.pretrainPredict(trainModel, test, f)

    def directPredict(self, trainModel):
        i = 0
        pretrainModel = self.getPretrainModel()
        outputs = list()
        for img, label, f in self.getImagesPatches(data_config.testData.dataFolder):
            s = time.time()
            feature = pretrainModel.predict(img, batch_size = 1)
            output = trainModel.predict(np.asarray([feature]), batch_size = 1)
            outputs.append([utils.getFileId(f), output[0][1]])
            print('processed images ETA {:.02f}s - {}'.format(time.time() -s, i), end='\r')

        return outputs

    def pretrainPredict(self, trainModel, data, files):
        output = trainModel.predict(data, batch_size = 32)
        return [ [files[i], output[i][1]] for i in range(len(files))]

    def debug(self, dataFolder):
        trainModel = self.getTrainModel()
        trainModel.load_weights(self.baseFolder + self.savedModel)
        data, lables, files = self.getPretrainDataPerFolder(dataFolder)
        output = trainModel.predict(data, batch_size = 32)
        pred  = np.argmax(output, axis = 1)
        expect = np.argmax(lables, axis = 1)
        incorrect = pred == expect
        return [ (files[i], expect[i], pred[i], output[i][1])
            for i, v in enumerate(incorrect) if v == False], self.baseFolder

    def getPretrainData(self):
        train = self.getPretrainDataPerFolder(data_config.trainData)
        train = train[:-1] #self.randomExpendPretrain(train, 10)
        validation = self.getPretrainDataPerFolder(data_config.validationData)
        return train, validation[:-1]

    def randomExpendPretrain(self, data, size):
        nxs = list()
        for x in data[0]:
            for _ in range(size):
                nx = np.take(x,np.random.permutation(x.shape[0]),axis=0)
                nxs.append(nx)

        nys = [np.tile(y, (size, 1)) for y in data[1]]

        return np.asarray(nxs), np.concatenate(nys)

    def pretrain(self):
        self.pretrainPerFolder(data_config.validationData)
        self.pretrainPerFolder(data_config.testData)
        self.pretrainPerFolder(data_config.trainData)

    def pretrainPerFolder(self, data):
        pretrainModel = self.getPretrainModel()
        features = list()
        labels = list()
        fileIds = list()
        i = 0
        for img, label, f in self.getImagesPatches(data.dataFolder):
            s = time.time()
            feature = pretrainModel.predict(img, batch_size = 1)
            features.append(feature)
            labels.append(label)
            fileIds.append(utils.getFileId(f))
            i += 1
            print('processed images ETA {:.02f}s - {}'.format(time.time() - s, i), end='\r')

        self.processPretrain(data, features, labels, fileIds)

    def processPretrain(self, data, features, labels, fileIds):
        utils.save_array(self.baseFolder + data.pretrain[0], features)
        utils.save_array(self.baseFolder + data.pretrain[1], np.concatenate(labels))
        utils.save_array(self.baseFolder + data.pretrain[2], fileIds)

    def getImagesPatches(self, path):
        batches = utils.getBatches(path, utils.invasive_image_size)
        for i in range(batches.samples):
            images, labels = batches.next()
            images = np.concatenate([
                images[:, x:x+utils.vgg_image_size[0], y:y+utils.vgg_image_size[1], :]
                for x in utils.patch_range[0] for y in utils.patch_range[1]])

            yield images, labels, batches.filenames[i]