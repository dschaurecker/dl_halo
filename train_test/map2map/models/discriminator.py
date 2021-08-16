from math import log2
import torch
import torch.nn as nn

from .narrow import narrow_by, narrow_like
from .resample import Resampler


class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """
    def __init__(self, cat, chan=1):
        super().__init__()

        self.cat = cat

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1])

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise
            x = x + noise

        return x

    
class D(nn.Module):
    def __init__(self, in_chan, out_chan, scale_factor=8, #scale_factor=16
                 chan_base=512, chan_min=64, chan_max=512):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            if b >= 0:
                c = chan_base >> b
            else:
                c = chan_base << -b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            nn.Conv3d(in_chan, chan(num_blocks), 1),
            nn.LeakyReLU(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            prev_chan, next_chan = chan(b+1), chan(b)
            self.blocks.append(ResBlock_D(prev_chan, next_chan))

        self.block9 = nn.Sequential(
            nn.Conv3d(chan(0), chan(-1), 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(chan(-1), 1, 1),
            #nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.block0(x)

        for block in self.blocks:
            x = block(x)

        x = self.block9(x)

        return x
    


    
class ResBlock_D(nn.Module):
    """The residual block of the StyleGAN2 discriminator.
    Downsampling are all linear, not strided convolution.
    Notes
    -----
    next_size = (prev_size - 4) // 2
    """
    def __init__(self, prev_chan, next_chan):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(prev_chan, prev_chan, 3),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(prev_chan, next_chan, 3),
            nn.LeakyReLU(0.2, True),
        )

        self.skip = nn.Conv3d(prev_chan, next_chan, 1)

        self.downsample = Resampler(3, 0.5)

    def forward(self, x):
        y = self.conv(x)

        x = self.skip(x)
        x = narrow_by(x, 2)

        x = x + y

        x = self.downsample(x)

        return x
    