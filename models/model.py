
from models.TransRef import TransRef
import torch


def create_model(opt):
    
    model = TransRef(opt = opt)
    
    print("model [%s] was created" % (model.name()))
    return model

