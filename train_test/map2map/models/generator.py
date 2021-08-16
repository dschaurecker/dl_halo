import torch
import torch.nn as nn
import pdb

from .conv import ConvBlock, ResBlock, ConvBlock_up, ResBlock_up
from .narrow import narrow_like, narrow_by

from math import log2

from .srsgan import AddNoise

    
class G(nn.Module):
    def __init__(self, in_chan, out_chan, bypass=None, **kwargs):
        """
        U-Net like network

        Note:

        Global bypass connection adding the input to the output (similar to
        COLA for displacement input and output) from Alvaro Sanchez Gonzalez.

        Global bypass, under additive symmetry, effectively obviates --aug-add
        """
        super().__init__()

        self.conv_l0 = ResBlock_up(in_chan, 64, seq='CANCA')
        self.down_l0 = ConvBlock_up(64, seq='NDA')
        self.conv_l1 = ResBlock_up(64, 128, seq='NCANCA')
        self.down_l1 = ConvBlock_up(128, seq='NDA')
        
        self.conv_c = ResBlock_up(128, seq='NCANCA')
        
        self.up_r1 = ConvBlock_up(128, seq='NUXANXA')
        self.conv_r1 = ResBlock_up(256, 64, seq='NCANCA')
        self.up_r0 = ConvBlock_up(64, seq='NUXANXA')
        self.conv_r0 = ConvBlock_up(128, out_chan, seq='CAC')

        self.bypass = False

    def forward(self, x):
        if self.bypass:
            x0 = x

        y0 = self.conv_l0(x)
        x = self.down_l0(y0)

        y1 = self.conv_l1(x)
        x = self.down_l1(y1)

        x = self.conv_c(x)

        x = self.up_r1(x)
        y1 = narrow_like(y1, x)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x)

        x = self.up_r0(x)
        y0 = narrow_like(y0, x)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r0(x)
        
        x = narrow_by(x, 1) 

        if self.bypass:
            x0 = narrow_like(x0, x)
            x += x0
        return x
    
    