from Models.VggFullPatchesModel import *
from Models.VggFullSampleModel import *
from Models.ResNet50PatchesModel import *

def getInvasiveModel():
    #return VggFullSampleModel()
    #return ResNet50PatchesModel()
    return VggFullPatchesModel()