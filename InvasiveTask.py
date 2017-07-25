import numpy as np
import sys
import os
import keras
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint
import InvasiveModelsFactory
import data_config
import utils
from Models.VggFullPatchesModel import *
from Models.VggFullSampleModel import *
from Models.ResNet50PatchesModel import *


def theano_config():
    import theano
    print(theano.config.device)

def tensorflow_config():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    print('tensorflow config')

def main():
    if keras.backend.backend() == 'theano':
        theano_config()
    elif keras.backend.backend() == 'tensorflow':
        tensorflow_config()

    model = InvasiveModelsFactory.getInvasiveModel()
    p = None if len(sys.argv) < 2 else sys.argv[1][0]
    if p == 'p':
        pretrain(model)
    elif p == 't':
        train(model, epochs = 1000, lr = 0.0001, batch_size = 16)
    elif p == 'e':
        predict(model, data_config.testData)
    elif p == 'd':
        debug(model)
    else:
        print('Do nothing')

def pretrain(model):
    model.pretrain()

def train(model, epochs = 200, lr = 0.01, batch_size = 16):
    trainModel = model.getTrainModel()
    trainModel.summary()
    trainModel.compile(optimizer = SGD(lr=lr, momentum=0.9), 
        loss = 'categorical_crossentropy', metrics=['accuracy'])

    (x, y), validation = model.getPretrainData()
    
    trainModel.fit(x, y, validation_data = validation,
        epochs = epochs, batch_size = batch_size, 
        callbacks=[ModelCheckpoint(model.modelSavePath, monitor='val_acc', save_best_only=True)])

    model.save()

def predict(model, data, modelPath = None):
    outputs = model.predict(data, modelPath)

    #outputs = [[o[0], np.clip(o[1], 0.05, 0.95)] for o in outputs]
    outputs = [ o[:-1] for o in outputs]
    model.saveSubmission(outputs)

    return outputs

def debug(model):
    incorrects, outputFolder = model.debug(data_config.validationData)
    print('total {}'.format(len(incorrects)))

    outputFolder = os.path.join(outputFolder, 'debug')
    utils.clear_or_mkdir(outputFolder)

    for i in incorrects:
        moveFile(i, data_config.validationData.dataFolder, outputFolder)


def moveFile(i, dataFolder, outputFolder):
    print('File {} from {}---> {}    ({}) '.format(i[0], i[1], i[2], i[3]))
    f = '{}/{}.jpg'.format(str(bool(i[1])).lower(), i[0])
    utils.copyfile(os.path.join(dataFolder, f), os.path.join(outputFolder,f))

if __name__ == '__main__':
    main()