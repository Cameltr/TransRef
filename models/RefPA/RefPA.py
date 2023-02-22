import torch.nn as nn
import torch

from models.RefPA.DeformableBlock import DeformableConvBlock
from util.util import showpatch
from models.RefPA.ChannelFusionAttention import PH

class RefPA(nn.Module):
    def __init__(self,in_channels):
        super(RefPA, self).__init__()
        self.PA = DeformableConvBlock(input_channels= in_channels)
       
        self.PH = PH(in_channels)
        #对齐了两次次
    def forward(self,ist_feature, rst_feature):

        st_out = self.deformblock(ist_feature, rst_feature) #输出aligned feature
        
        out = self.fusion(ist_feature,st_out)
        
        return out 


