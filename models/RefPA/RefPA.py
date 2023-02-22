import torch.nn as nn
import torch

from models.RefPA.PA import PA
from util.util import showpatch
from models.RefPA.PH import PH

class RefPA(nn.Module):
    def __init__(self,in_channels):
        super(RefPA, self).__init__()
        self.PA = PA(input_channels= in_channels)
       
        self.PH = PH(in_channels)
        #对齐了两次次
    def forward(self,input_feature, ref_feature):

        coarse_out = self.PA(input_feature, ref_feature) 
        
        out = self.PH(input_feature,coarse_out)
        
        return out 


