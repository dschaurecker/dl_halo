from math import log2, log10, ceil
import warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable

from ..models import lag2eul, power

def fig3d_zoom(*fields, path=None, size=32, cmap=None, norm=None):
    fields = [field.detach().cpu().numpy() if isinstance(field, torch.Tensor)
            else field for field in fields]
    
    assert all(isinstance(field, np.ndarray) for field in fields)

    nc = max(field.shape[0] for field in fields)
    nf = len(fields)
    
    colorbar_frac = 0.15 / (0.85 * nc + 0.15) #size of colorbar
    fig, axes = plt.subplots(nc, nf, squeeze=False,
            figsize=(4 * nf, 4 * nc * (1 + colorbar_frac)))

    fig.suptitle(path)
    
    def quantize(x):
        return 2 ** round(log2(x), ndigits=1)

    for f, field in enumerate(fields): 
        
        
        for c in range(field.shape[0]):
            im = axes[c, f].imshow(np.sum(field[c, :, :, :], axis=1)) #project on x axis
            #print('pltslice: min, max:', field[c, :, :, :].min(), field[c, :, :, :].max())
            plt.colorbar(im, ax=axes[:, f])
        for c in range(field.shape[0], nc):
            axes[c, f].axis('off')
            
        if f == 0: axes[c, f].set_title('Input')
        elif f == 1: axes[c, f].set_title('Prediction')
        elif f == 2: axes[c, f].set_title('Target')
        elif f == 3: axes[c, f].set_title('Difference')


    return fig


def plt_power(*fields, l2e=False, label=None):
    """Plot power spectra of fields. They are unnormed so input and target power will have different amplitudes.
    Each field should have batch and channel dimensions followed by spatial
    dimensions.
    Optionally the field can be transformed by lag2eul first.
    See `map2map.models.power`.
    """
    plt.close('all')

    if label is not None:
        assert len(label) == len(fields)
    else:
        label = [None] * len(fields)

    with torch.no_grad():
        if l2e:
            fields = lag2eul(*fields)

        ks, Ps = [], []
        for field in fields:
            k, P, _ = power(field)
            ks.append(k)
            Ps.append(P)

    ks = [k.cpu().numpy() for k in ks]
    Ps = [P.cpu().numpy() for P in Ps]

    fig, axes = plt.subplots(figsize=(4.8, 3.6), dpi=150)

    for k, P, l in zip(ks, Ps, label):
        axes.loglog(k, P, label=l, alpha=0.7)

    axes.legend()
    axes.set_xlabel('unnormalized wavenumber')
    axes.set_ylabel('unnormalized power')

    fig.tight_layout()

    return fig


def plt_hist(*fields, intype=None, boxsize=None, label=None):
    """Plot histogram of fields
    """
    plt.close('all')
    
    intp = intype
    boxs = boxsize
    
    with torch.no_grad():
    
        if label is not None:
            assert len(label) == len(fields)
        else:
            label = [None] * len(fields)

        bins = 30

        fig, ax = plt.subplots(1, 1, figsize=(10,7))

        ax.set_title('number count values)')
        for (field, lbl) in zip(fields, label):
            #print('label: ',lbl)
            #print('plt_hist field',field)
            ax.hist(field.cpu().numpy().flatten(), label=lbl, bins=bins, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')
        ax.set_ylabel('# of voxels')
        ax.set_xlabel('unnormed count value')

    return fig