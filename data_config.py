import os

class DataDivision:
    def __init__(self, dataFolder, pretrain):
        self.dataFolder = dataFolder
        self.pretrain = (pretrain, pretrain + '_lable', pretrain + '_id')

trainData = DataDivision('data/train', 'pretrain')
validationData = DataDivision('data/validation', 'prevalidation')
testData = DataDivision('data/test', 'pretest')

label_path = 'data/train_labels.csv'
image_postfix = '.jpg'

true_folder = 'true'
false_folder = 'false'