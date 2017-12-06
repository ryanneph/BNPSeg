import sys
import os
import numbers
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import logging
from pymedimage.misc import ensure_extension
from pymedimage.rttypes import MaskableVolume


logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING) # suppress PIL logging

def loadImageSet(dname, visualize=False, ftype='float32', normalize=True, resize=None):
    """load a set of rgb images conatained within a single directory's top level
    Each image is loaded using PIL as an rgb image then linearized into a matrix [shape=(N,D)] containing N
    pixels in row-major (zyx) order, and D-dimensional pixel appearance features

    Args:
        dname (str): path to image collection containing directory
    """
    ims = []
    sizes = []
    fnames = []
    common_dim = None
    for fname in sorted([os.path.join(dname, x) for x in os.listdir(dname) if os.path.isfile(os.path.join(dname,x))]):
        with open(fname, 'rb') as f:
            im = Image.open(f, 'r')
            # linearize array
            if im.mode in ['1', 'L', 'P']:
                dim = 1
                if im.mode=='P':
                    im = im.convert('L')
            elif im.mode in ['RGB', 'YCbCr']:
                dim = 3
            elif im.mode in ['RGBA', 'CMYK']:
                dim = 4
            else:
                raise RuntimeError("Couldn't determine dimensionality of image with mode=\"{!s}\"".format(im.mode))
            maxint = 255 # assume all 8-bit per channel

            if common_dim is None:
                common_dim = dim
            elif dim != common_dim:
                if common_dim == 1:
                    # convert to grayscale
                    logger.warning('image: {} has been converted from "{}" to "{}"'.format(fname, im.mode, "L"))
                    im = im.convert('L')
                else:
                    raise RuntimeError(("Dimensionality of image: \"{}\" (\"{}\":{})" +
                                       "doesn't match dataset dimensionality ({})").format(
                                           fname, im.mode, dim, common_dim))

            # resize image
            if isinstance(resize, numbers.Number) and resize>0 and not resize==1:
                im = im.resize( [int(resize*s) for s in im.size] )

            # reshape into matrix of shape=(N,dim) with the first axis in row-major order
            arr = np.array(im).reshape(-1, common_dim).astype(ftype)
            if normalize:
                # normalize to [0,1]
                for i in range(common_dim):
                    arr[:,i] = arr[:,i] / maxint

            if visualize:
                tempim = arr.reshape(*im.size[::-1], common_dim)
                plotChannels(tempim)
                plt.show()
            ims.append(arr)
            sizes.append(im.size[::-1])
            fnames.append(fname)
            logger.debug('loaded image: {}, (h,w)={}, shape=({})'.format(fname, sizes[-1], arr.shape))
    return ims, sizes, fnames, common_dim

def plotChannels(arr):
    fig = plt.figure(figsize=(9,3))
    titles = ['red', 'green', 'blue']
    for i in range(arr.shape[-1]):
        ax = fig.add_subplot(1,arr.shape[-1],i+1)
        ax.imshow(arr[:,:,i], cmap="Greys")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title(titles[i])
    return fig

def saveMosaic(collection, fname, figsize=(10,10), cmap="Set3", header=None, footer=None):
    if cmap is None and collection[0].shape[-1] == 1:
        cmap="Greys"

    # plotting parameters
    spacing = 0.01
    margin  = 0.025
    headerheight = 0.04 * int(bool(header))
    footerheight = 0.04 * int(bool(footer))

    # add header/footer text
    fig = plt.figure(figsize=figsize)
    if header is not None:
        fig.text(0.5, 1-(margin+0.5*headerheight), str(header),
                 horizontalalignment='center', verticalalignment='bottom', transform=fig.transFigure)
    if footer is not None:
        fig.text(0.5, (margin+0.5*footerheight), str(footer),
                 horizontalalignment='center', verticalalignment='top', transform=fig.transFigure)

    # construct mosaic
    Nj = len(collection)
    nrow = math.ceil(math.sqrt(Nj))
    ncol = nrow - (1 if nrow*(nrow-1)>=Nj else 0)
    wper = (1-2*margin-(ncol-1)*spacing)/ncol
    hper = (1-2*margin-headerheight-footerheight-(nrow-1)*spacing)/nrow
    for j in range(Nj):
        yy = math.floor(j/ncol)
        xx = j % ncol
        ax = fig.add_axes([xx*wper+xx*spacing+margin,
                           1-(yy*hper+yy*spacing+margin+headerheight)-hper,
                           wper, hper])
        ax.imshow(np.squeeze(collection[j]), cmap=cmap, interpolation=None)
        #  ax.set_axis_off()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    fig.savefig(fname)
    plt.close(fig)

def saveImage(array, fname, mode='L', resize=None, cmap='Set3'):
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError('Saving image with ndim={} is not supported'.format(array.ndim))

    if mode in ['RGB', 'RGBA']:
        # convert integer class ids to rgb colors according to cmap
        rng = abs(np.max(array)-np.min(array))
        if rng == 0: rng = 1
        normarray = (array - np.min(array)) / rng
        im = Image.fromarray(np.uint8(plt.cm.get_cmap(cmap)(normarray)*255))
    elif mode in ['P']:
        # separates gray values so they can be distinguished
        array*=math.floor((255 / len(np.unique(array))))
        im = Image.fromarray(array.astype('uint8'))
    elif mode in ['1', 'L', 'P']:
        im = Image.fromarray(array.astype('uint8'))
    else: raise RuntimeError

    # restore image to original dims
    if isinstance(resize, numbers.Number) and resize>0 and not resize==1:
        im = im.resize( [int(resize*s) for s in im.size], resample=Image.NEAREST)

    fname = ensure_extension(fname, '.png')
    im.save(fname)
    logger.debug('file saved to {}'.format(fname))
