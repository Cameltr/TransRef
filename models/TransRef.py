from hashlib import algorithms_guaranteed
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import warnings

from models.base_networks import *
from models import networks
from models.base_model import BaseModel
from util.util import showpatch
from models.loss import VGG16, PerceptualLoss, StyleLoss
from models.RefPA.RefPA  import RefPA

class EncoderTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=6, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.patch_embed1_ref = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
                                              embed_dim=embed_dims[0])
        self.patch_embed2_ref = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3_ref = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4_ref = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        # for Intra-patch transformer blocks

        self.mini_patch_embed1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                   in_chans=embed_dims[0],
                                                   embed_dim=embed_dims[1])
        self.mini_patch_embed1_ref = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                   in_chans=embed_dims[0],
                                                   embed_dim=embed_dims[1])
        self.mini_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                   in_chans=embed_dims[1],
                                                   embed_dim=embed_dims[2])
        self.mini_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                   in_chans=embed_dims[2],
                                                   embed_dim=embed_dims[3])
        self.mini_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 32, patch_size=3, stride=2,
                                                   in_chans=embed_dims[0],
                                                   embed_dim=embed_dims[3])

        self.mini_patch_embed2_ref = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                   in_chans=embed_dims[1],
                                                   embed_dim=embed_dims[2])
        self.mini_patch_embed3_ref = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                   in_chans=embed_dims[2],
                                                   embed_dim=embed_dims[3])
        self.mini_patch_embed4_ref = OverlapPatchEmbed(img_size=img_size // 32, patch_size=3, stride=2,
                                                   in_chans=embed_dims[0],
                                                   embed_dim=embed_dims[3])

        self.RefPA1 = RefPA(embed_dims[0])
        self.RefPA2 = RefPA(embed_dims[1])
        self.RefPA3 = RefPA(embed_dims[2])

        # main  encoder1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        # intra-patch encoder1
        self.patch_block1 = nn.ModuleList([Block_Ref(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(1)])
        self.pnorm1 = norm_layer(embed_dims[1])
        # main  encoder2
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        # intra-patch encoder2
        self.patch_block2 = nn.ModuleList([Block_Ref(
            dim=embed_dims[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1)])
        self.pnorm2 = norm_layer(embed_dims[2])
        # main  encoder3
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        # intra-patch encoder3
        self.patch_block3 = nn.ModuleList([Block_Ref(
            dim=embed_dims[3], num_heads=num_heads[1], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1)])
        self.pnorm3 = norm_layer(embed_dims[3])
        # main  encoder4
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

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

    def init_weights(self, pretrained=None):
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x, ref):
        B = x.shape[0] # 
        outs = []
        embed_dims = [64, 128, 320, 512]
        ## stage 1
        
        x1, H1, W1 = self.patch_embed1(x) #x [1, 6, 256, 256] x1 [1, 4096, 64]

        x1_ref,H_ref,W_ref  = self.patch_embed1_ref(ref) #ref [1, 3, 256, 256], x1_ref [1, 4096, 64]
        
    
        # input ref 大特征对齐融合 输出a1 #64
        a1 = self.RefPA1(x1.permute(0, 2, 1).reshape(B, embed_dims[0], H1, W1).contiguous(),x1_ref.permute(0, 2, 1).reshape(B, embed_dims[0], H_ref, W_ref).contiguous())
        
        # 对ref 进行mini-embedding
        x2_ref, H2_ref, W2_ref = self.mini_patch_embed1_ref(x1_ref.permute(0, 2, 1).reshape(B, embed_dims[0], H_ref,W_ref))
        
        # 对a1进行小的embedding 
        x2, H2, W2 = self.mini_patch_embed1(a1)

        # TB计算
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous() #x1 [1 64 64 64]

        # mini计算
        for i, blk in enumerate(self.patch_block1):
            x2 = blk(x2, H2, W2, x2_ref)
        x2 = self.pnorm1(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()  #x2 [1 128 32 32]
        
        x2_interpolate = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        
        outs.append(x1)

        ## stage 2
        x1, H1, W1 = self.patch_embed2(x1) #1 64 64 64  #1 1024 128
        
        x1_ref = x1_ref.permute(0, 2, 1).reshape(B, embed_dims[0], H_ref, W_ref) #1, 64, 64, 64  #1, 4096, 64
        
        x1_ref, H_ref, W_ref = self.patch_embed2_ref(x1_ref) #1 1024 128

        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[1], H1, W1) + x2  #1 128 32 32

        # 32
        a2 = self.RefPA2(x1.contiguous(),x1_ref.permute(0, 2, 1).reshape(B, embed_dims[1], H_ref, W_ref).contiguous())

        x2_ref, H2_ref, W2_ref = self.mini_patch_embed2_ref(x1_ref.permute(0, 2, 1).reshape(B, embed_dims[1], H_ref, W_ref))

        x2, H2, W2 = self.mini_patch_embed2(a2)
        
        x1_ref = x1_ref.reshape(B, H_ref, W_ref, -1).permute(0, 3, 1, 2).contiguous()

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous() # [1, 128, 32, 32]
        outs.append(x1)

        
        for i, blk in enumerate(self.patch_block2):
            x2 = blk(x2, H2, W2, x2_ref)
        x2 = self.pnorm2(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous() # [1, 320, 16, 16]
        
        x2_interpolate = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1) #1 256 320, #1 128 32 32
  
        x1_ref, H_ref, W_ref = self.patch_embed3_ref(x1_ref) #1 256 320, #1 128 32 32

        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[2], H1, W1) + x2 # #1 320 16 16

        # 16
        a3 = self.RefPA3(x1.contiguous(),x1_ref.permute(0, 2, 1).reshape(B, embed_dims[2], H_ref, W_ref).contiguous())

        x2, H2, W2 = self.mini_patch_embed3(a3)
        x2_ref, H2_ref, W2_ref = self.mini_patch_embed3_ref(x1_ref.permute(0, 2, 1).reshape(B, embed_dims[2], H_ref, W_ref))

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        

        for i, blk in enumerate(self.patch_block3):
            x2 = blk(x2, H2, W2,x2_ref)
        x2 = self.pnorm3(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous() # [1, 512, 8, 8]
        
        x2_interpolate = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        
        # stage 4
        x1, H1, W1 = self.patch_embed4(x1) #[1, 64, 512], #[1, 320, 16, 16]
        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[3], H1, W1) + x2 # [1, 512, 8, 8]

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)#[1, 64, 512]

        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous() # [1, 512, 8, 8]
        outs.append(x1)

        return outs

    def forward(self, x,ref):
        x = self.forward_features(x,ref)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x):
        # pdb.set_trace()
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


############################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_Ref(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W,ref):

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        print(q.shape)
        if self.sr_ratio > 1:
            ref_ = ref.permute(0, 2, 1).reshape(B, C, H, W)
            ref_ = self.sr(ref_).reshape(B, C, -1).permute(0, 2, 1)
            ref_ = self.norm(ref_)
            kv = self.kv(ref_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(ref).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        print(k.shape)
        print(v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_vis = x.transpose(-2, -1).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W):

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1, 48, dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W):

        B, N, C = x.shape
        task_q = self.task_query

        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# Transformer decoder block
class Block_dec(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

# main-PT block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# Ref-PT block
class Block_Ref(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_Ref = norm_layer(dim)
        self.attn = Attention_Ref(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W, ref):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W,self.norm1_Ref(ref)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DecoderTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[3])

        # transformer decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block_dec(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[3])

        cur += depths[0]

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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            pass
            #logger = get_root_logger()
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        x = x[3]
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class Tenc(EncoderTransformer):
    def __init__(self, **kwargs):
        super(Tenc, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 4], mlp_ratios=[2, 2, 2, 2],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[4, 2, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class Tdec(DecoderTransformer):
    def __init__(self, **kwargs):
        super(Tdec, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class convprojection(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection, self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()

    def forward(self, x1, x2):

        res32x = self.convd32x(x2[0])

        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, -1, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)

        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0, -1, 0, 0)
            res32x = F.pad(res32x, p2d, "constant", 0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, 0, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)

        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x)

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, -1, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0, -1, 0, 0)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, 0, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x)
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x


## The following is the network which can be fine-tuned for specific datasets
class TransRef_Base(nn.Module):

    def __init__(self, path=None, **kwargs):
        super(TransRef_Base, self).__init__()

        self.Tenc = Tenc()

        self.Tdec = Tdec()

        self.convtail = convprojection()

        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()

        if path is not None:
            self.load(path)

    def forward(self, x,ref):
    
        #ref = torch.randn(1, 3, 256, 256).cuda()
        
        x1 = self.Tenc(x,ref)

        x2 = self.Tdec(x1)

        x = self.convtail(x1, x2)

        clean = self.active(self.clean(x))

        return clean

    def load(self, path):
        """
        Load checkpoint.
        """
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        model_state_dict_keys = self.state_dict().keys()
        #checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['state_dict'], "module.")
        #self.load_state_dict(checkpoint_state_dict_noprefix, strict=False)
        del checkpoint
        torch.cuda.empty_cache()


class TransRef(BaseModel):

    def __init__(self, opt=None):
        super(TransRef, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.ref = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_img = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_mask = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        self.device = torch.device('cuda')

        self.model =  TransRef_Base().to(self.device)

        self.vgg = VGG16()
        self.PerceptualLoss = PerceptualLoss()
        self.StyleLoss = StyleLoss()
        self.model_names = ['Trans']
        self.criterionL1 = torch.nn.L1Loss()
        if self.isTrain:
            self.schedulers = []
            self.optimizers = []
            self.optimizer_model = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_model)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
            print('---------- Networks initialized -------------')

        if self.isTrain:
            if opt.continue_train :
                print('Loading pre-trained network!')
                self.load_networks(self.model, 'Trans', opt.which_epoch)

    def name(self):
        return self.modelname
    
    def mask_process(self, mask):
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()
        return mask
    
    def set_input(self, input_De, input_Mask, ref_De):
        self.Gt_DE = input_De.to(self.device)
        
        self.input_DE = input_De.to(self.device)
        self.ref_DE = ref_De.to(self.device)
        

        self.input_mask_global = self.mask_process(input_Mask.to(self.device))

        self.Gt_Local = input_De.to(self.device)
        # define local area which send to the local discriminator
        self.crop_x = random.randint(0, 191)
        self.crop_y = random.randint(0, 191)
        self.Gt_Local = self.Gt_Local[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        
        self.ex_input_mask = self.input_mask_global.expand(self.input_mask_global.size(0), 3,
                                                           self.input_mask_global.size(2),
                                                           self.input_mask_global.size(3))

        # unpositve with original mask

        self.inv_ex_input_mask = torch.add(torch.neg(self.ex_input_mask.float()), 1).float()
        self.inv_ex_input_mask = self.inv_ex_input_mask.repeat(input_Mask.shape[0],1,1,1)


        # Do not set the mask regions as 0
        self.input_DE.narrow(1, 0, 1).masked_fill_(self.input_mask_global.narrow(1, 0, 1).bool(),
                                                   2 * 123.0 / 255.0 - 1.0)
        self.input_DE.narrow(1, 1, 1).masked_fill_(self.input_mask_global.narrow(1, 0, 1).bool(),
                                                   2 * 104.0 / 255.0 - 1.0)
        self.input_DE.narrow(1, 2, 1).masked_fill_(self.input_mask_global.narrow(1, 0, 1).bool(),
                                                   2 * 117.0 / 255.0 - 1.0)


    def forward(self):
        
        input = torch.cat([self.input_DE,self.inv_ex_input_mask], 1)
        
        self.out = self.model(input,self.ref_DE)

    def backward(self):
        self.loss_L1 = self.criterionL1(self.out, self.Gt_DE)
        self.Perceptual_loss = self.PerceptualLoss(self.out, self.Gt_DE)
        self.Style_Loss = self.StyleLoss(self.out, self.Gt_DE)
        self.loss = self.loss_L1 * self.opt.lambda_L1 + self.Perceptual_loss * self.opt.lambda_P + self.Style_Loss * self.opt.lambda_S
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad(self.model, True)
        self.optimizer_model.zero_grad()
        self.backward()
        self.optimizer_model.step()

    def get_current_errors(self):
        # show the current loss

        return OrderedDict([('loss_L1', self.loss_L1.data),
                            ('Perceptual_loss', self.Perceptual_loss.data),
                            ('Style_loss', self.Style_Loss.data)
                            ])

    def get_current_visuals(self):
        input_image = (self.input_DE.data.cpu() + 1) / 2.0
        ref_image = (self.ref_DE.data.cpu() + 1) / 2.0
        
        fake_image = (self.out.data.cpu() + 1) / 2.0
        real_gt = (self.Gt_DE.data.cpu() + 1) / 2.0
        return input_image, ref_image, fake_image, real_gt

    def load(self, path):
        """
        Load checkpoint.
        """
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        # weights_dict = {}
        # for k, v in checkpoint['state_dict'].items():
        #     if 'backbone' in k:
        #         new_k = k.replace('backbone.', 'encoder.')
        #         weights_dict[new_k] = v

        # self.load_state_dict(weights_dict)
        model_state_dict_keys = self.state_dict().keys()
        #checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['state_dict'], "module.")
        #self.load_state_dict(checkpoint_state_dict_noprefix, strict=False)
        del checkpoint
        torch.cuda.empty_cache()


