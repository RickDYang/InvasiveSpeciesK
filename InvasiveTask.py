import numpy as np
import sys
import os
from keras.optimizers import Adam, RMSprop, SGD
import InvasiveModelsFactory
import data_config
import utils
import theano
print(theano.config.device)

def main():
    model = InvasiveModelsFactory.getInvasiveModel()
    p = None if len(sys.argv) < 2 else sys.argv[1][0]
    if p == 'p':
        pretrain(model)
    elif p == 't':
        train(model, 100)
    elif p == 'e':
        predict(model)
    elif p == 'd':
        debug(model)
    else:
        print('Do nothing')

def pretrain(model):
    model.pretrain()

def train(model, epochs = 200, lr = 0.1, batch_size = 64):
    trainModel = model.getTrainModel()
    trainModel.summary()
    trainModel.compile(optimizer = Adam(lr=lr, decay=lr/(epochs * 0.9999)), 
        loss = 'categorical_crossentropy', metrics=['accuracy'])

    (x, y), validation = model.getPretrainData()

    trainModel.fit(x, y, validation_data = validation,
        epochs = epochs, batch_size = batch_size)

    model.save()

def predict(model, modelPath = None):
    outputs = model.predict(modelPath)

    #outputs = [[o[0], np.clip(o[1], 0.05, 0.95)] for o in outputs]
    model.saveSubmission(outputs)

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