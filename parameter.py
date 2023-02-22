
import torch
from thop import profile
from models.TransRef import TransRef_Base
model = TransRef_Base().cuda()
# Model
print('==> Building model..')

dummy_input = torch.randn(1, 6, 256, 256).cuda()
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))