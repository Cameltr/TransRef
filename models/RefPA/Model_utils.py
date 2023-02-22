import torch
import torch.nn as nn


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class residual_block(nn.Module):
    def __init__(self, input_channel = 256, output_channel = 256, bias = False):
        super(residual_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=input_channel, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel, kernel_size=3, padding=1, bias = bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
     #  out *= 0.1 for bigmodel
        out = torch.add(out,x)

        return out

def make_residual_block(blocknum=32, input_channel = 64, output_channel = 64, bias = False):
    residual_layers = []
    #residual_layers.append(residual_block(input_channel=input_channel, output_channel = output_channel,bias=bias))
    for i in range(blocknum):
        residual_layers.append(residual_block(input_channel=output_channel, output_channel = output_channel, bias = bias))
    blockpart_model = nn.Sequential(*residual_layers)
    return blockpart_model

def make_downsampling_network(layernum = 2, in_channels = 3, out_channels = 64):
    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False, padding=1))
    for _ in range(layernum-1):
        layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False,padding=1))
    print(layers)
    model = nn.Sequential(*layers)
    return model

def DOE_downsample_block(input_channelsize):
    layers = []
    layers.append(
        nn.Conv2d(in_channels=input_channelsize, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True))
    layers.append(nn.LeakyReLU(inplace=True))

    pre_model = nn.Sequential(*layers)
    return pre_model

def DOE_upsample_block(in_odd = True, in_channels = 64, out_channels = 64):
    layers = []

    if in_odd:
        layers.append(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=True))
    else:
        layers.append(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=0,
                               bias=True))
    layers.append(nn.LeakyReLU(inplace=True))

    post_model = nn.Sequential(*layers)
    return post_model