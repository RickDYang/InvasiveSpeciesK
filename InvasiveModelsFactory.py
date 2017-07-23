from Models.VggFullPatchesModel import *
from Models.VggFullSampleModel import *
from Models.ResNet50PatchesModel import *
from Models.InvasiveCombineModel import *
from Models.VggNoTopSampleModel import *
from Models.VggNotopPatchesModel import *

def getInvasiveModel():
    return VggNotopPatchesModel()
    #return VggFullSampleModel()
    #return VggNoTopSampleModel()
    #return ResNet50PatchesModel()
    #return VggFullPatchesModel()
    #return InvasiveCombineModel([VggFullPatchesModel(), ResNet50PatchesModel()])