import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch


class PH(nn.Module):
    def __init__(self, input_channels, reduction=8):
        super(PH, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(input_channels*2, input_channels // reduction, kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(input_channels // reduction, input_channels*2, kernel_size = 1),
            nn.GELU()
        )
        self.reduc = nn.Sequential(
            nn.Conv2d(input_channels*2, input_channels, kernel_size = 1),nn.GELU()
        
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x,ref):
        input = torch.cat([x, ref], 1)
        #b, c, _, _ = input.size()
        #y = self.avg_pool(x).view(b, c)
        y = self.fc(input)
        res = input * y
        res = self.reduc(res)
        return res

