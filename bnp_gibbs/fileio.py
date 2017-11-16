import sys
import os
import numbers
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from loggers import RotatingFile

def loadImageSet(dname, verbose=False, interactive=False, ftype='float32', normalize=True, resize=None):
    """load a set of rgb images conatained within a single directory's top level
    Each image is loaded using PIL as an rgb image then linearized into a matrix [shape=(N,D)] containing N
    pixels in row-major (zyx) order, and D-dimensional pixel appearance features

    Args:
        dname (str): path to image collection containing directory
    """
    ims = []
    sizes = []
    common_dim = None
    for fname in [os.path.join(dname, x) for x in os.listdir(dname) if os.path.isfile(os.path.join(dname,x))]:
        with open(fname, 'rb') as f:
            im = Image.open(f, 'r')
            # linearize array
            if im.mode in ['1', 'L', 'P']:
                dim = 1
            elif im.mode in ['RGB', 'YCbCr']:
                dim = 3
            elif im.mode in ['RGBA', 'CMYK']:
                dim = 4
            else:
                raise RuntimeError(f"Couldn't determine dimensionality of image with mode=\"{im.mode!s}\"")
            maxint = 255 # assume all 8-bit per channel

            if common_dim is None:
                common_dim = dim
            elif dim != common_dim:
                raise RuntimeError(f"Dimensionality of image: \"{fname}\" (\"{im.mode}\":{dim}) \
                                   doesn't match dataset dimensionality ({common_dim})")

            # resize image
            if isinstance(resize, numbers.Number) and resize>0:
                im = im.resize( [int(resize*s) for s in im.size] )

            arr = np.rollaxis(np.array(im), 2).reshape((dim, -1)).T.astype(ftype)
            if normalize:
                # normalize to [0,1]
                for i in range(dim):
                    arr[:,i] = arr[:,i] / maxint

            if interactive:
                plotRGB(np.moveaxis(arr.T.reshape((dim, *im.size[::-1])), 0, 2))
                plt.show()
            ims.append(arr)
            sizes.append(im.size[::-1])
            if verbose: print(f'loaded image: {fname}, (h,w)={im.size}, shape=({arr.shape})')
    return ims, sizes, common_dim

def plotRGB(arr):
    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(arr[:,:,0], cmap="Greys")
    ax.set_title('red')
    ax = fig.add_subplot(1,3,2)
    ax.imshow(arr[:,:,1], cmap="Greys")
    ax.set_title('green')
    ax = fig.add_subplot(1,3,3)
    ax.imshow(arr[:,:,2], cmap="Greys")
    ax.set_title('blue')
    return fig

def savefigure(collection, fname, factor=0.5):
    fig = plt.figure(figsize=(factor*16,factor*9))
    Nj = len(collection)
    nrow = math.ceil(math.sqrt(Nj))
    ncol = nrow - (1 if nrow*(nrow-1)>=Nj else 0)
    for j in range(Nj):
        ax = fig.add_subplot(nrow, ncol, j+1)
        ax.imshow(collection[j], cmap="Set3", interpolation=None)

    fig.savefig(fname)
    plt.close(fig)
