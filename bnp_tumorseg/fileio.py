import sys
import os
import numbers
import math
import numpy as np
import numpy.ma as ma
from PIL import Image
from matplotlib import pyplot as plt
import logging
from pymedimage.misc import ensure_extension
from pymedimage.rttypes import MaskableVolume
from pymedimage.fileio.general import loadImageCollection
from pymedimage.fileio.common_naming import gettype_BRATS17
from pymedimage.visualgui import multi_slice_viewer as view3d
import re

def gettype_rtfeature(fname):
    m = re.search(r'feature=(\w+)_args', fname)
    if m is not None: return m.group(1)
    else: return None

logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING) # suppress PIL logging
logging.getLogger('pymedimage').setLevel(logging.DEBUG)

def loadImageSet(dname, **kwargs):
    try:
        result = loadImageSet_natural(dname, **kwargs)
        if result: return result
    except: pass

    result = loadImageSet_medical(dname, **kwargs)
    if result: return result

    raise RuntimeError('Failed to load images at "{}"'.format(dname))

def loadImageSet_medical(dname, visualize=False, ftype='float32', normalize=True, resize=None):
    """recursively walk within dname, loading each directory of images as a separate 'document' with mulitple
    channels"""
    ims = []
    sizes = []
    fnames = []
    d = loadImageCollection(dname,
                            exts=['.nii', '.nii.gz', '.mha', '.h5'],
                            #  type_order=['t1','t1ce','t2','flair'],
                            #  type_order=['fo_energy', 'fo_entropy', 'fo_kurtosis', 'fo_rms', 'fo_stddev',
                            #              'fo_uniformity','fo_variance', 'glcm_clustertendency_min',
                            #              'glcm_contrast_min'],
                            multichannel=True,
                            typegetter=gettype_rtfeature,
                            #  typegetter=gettype_BRATS17,
                            asarray=True,
                            resize_factor=resize)
    dim = next(iter(d.values())).shape[0]
    for _f, _im in d.items():
        startslice = 0
        nslices = _im.shape[1]
        _im_reshaped = _im[:,startslice:startslice+nslices,:,:].reshape(dim, nslices, -1).T.astype(ftype)
        for sl in range(nslices):
            ims.append(_im_reshaped[:, sl, :])
            sizes.append(_im.shape[2:])
            fnames.append(os.path.join(dname, _f+str(sl+startslice)))
    masks = None # TODO: implement masking
    logger.debug('loaded {} slices from image: {}, (h,w)={}, shape={}'.format(nslices, fnames[-1], sizes[-1], ims[-1].shape))
    if ims: return ims, masks, sizes, fnames, dim
    else:   return None

def loadImageSet_natural(dname, visualize=False, ftype='float32', normalize=True, resize=None):
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
    if ims: return ims, None, sizes, fnames, common_dim
    else:   return None

def mask(collection, masks=None, maskval=None):
    """mask values by pruning out values according to maskcollection or equality to maskval"""
    if masks is not None:
        maskval = None
    elif maskval is not None:
        masks = []
        for im in collection:
            masks.append(np.where(im[:,0]>maskval, 1, 0))
    else: return (collection, [None]*len(collection))

    masked = []
    for im, mask in zip(collection, masks):
        masked.append( np.atleast_1d( im[mask!=0, :] ) )
    return masked, masks

def unmask(collection, masks=None, channels=1, fill_value=0):
    """use masks to expand pruned vectors back into properly shaped dense representations"""
    if masks is None: return collection
    if not isinstance(collection, list): collection = [collection]
    if not isinstance(masks, list): masks = [masks]
    full = []

    for vec, mask in zip(collection, masks):
        _im = fill_value*np.ones((*mask.shape, channels), dtype=float)
        _im[np.nonzero(mask), :] = vec
        full.append(np.atleast_1d(_im))
    if len(full) == 1: return full[0]
    else: return full

def normalize(collection):
    """normalize a set of arrays by scaling the sample max to 1 and sample min to 0 for each channel
    independently

    Args:
        collection (list): list of (m, d) or (m, 1) arrays where m is # pixels in image, d is # channels
    """
    if isinstance(collection, list) and len(collection) <= 0:
        raise RuntimeError('Collection provided was empty')
    if not isinstance(collection, list): collection = [collection]
    dim = collection[0].shape[-1]
    for d in range(dim):
        smin = 9999.0
        smax = -9999.0
        for im in collection:
            _min = np.min(im[:,d])
            _max = np.max(im[:,d])
            if _min < smin: smin = _min
            if _max > smax: smax = _max
        logger.debug3('channel #{}: min={}, max={}'.format(d, smin, smax))

        for im in collection:
            im[:,d] = (im[:,d] - smin) / (smax-smin)
    if len(collection) == 1: return collection[0]
    else: return collection


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

def splitSlices(collection):
    """if data is multichannel, split each channel into separate slice and package into a list for each item
    in the collection. does not support multichannel volumetric data

    Returns: [images...] for single channel (2d) data, or [ [slices], [slices]... ] for multichannel images"""
    newcollection = []
    for c in collection:
        slices = []
        if c.ndim == 3:
            for s in range(c.shape[-1]):
                slices.append(np.squeeze(c[:,:,s]))
        elif c.ndim > 3: raise RuntimeError('expected ndim <=3, not {}'.format(c.ndim))
        else: slices.append(np.squeeze(c))
        newcollection.append(slices)
    return newcollection

def saveMosaic(slices, fname, figsize=(10,10), cmap="Set3", header=None, footer=None, **kwargs):
    """save a tiling of each image in the collection. if each item in the collection is of ndim>2 then
    save a tiling of the channels in each item to a separate mosaic instead

    collection is either [ [slice1, slice2...], [slice1, slice2...] ] or [slice1, slice2...] (see splitSlices())
    """
    plotslices = []
    fnames = []
    for i, s in enumerate(slices):
        if not isinstance(s, list):
            plotslices.append(slices)
            fnames.append(fname)
            break
        else:
            base, ext = os.path.splitext(fname)
            for n in range(len(s)):
                f = '{}_im{}_sl{}.{}'.format(base, i, n, ext)
                fnames.append(f)
            plotslices.append(s)

    for c, _f in zip(plotslices, fnames):
        for slice in c:
            if slice.ndim > 2: raise RuntimeError('dimensionality of each item is expected to be 2, not {}'.format(slice.ndim))
        if cmap is None and c[0].ndim <= 2: cmap="gray"

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
        Nj = len(c)
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
            ax.imshow(c[j], cmap=cmap, interpolation=None, **kwargs)
            #  ax.set_axis_off()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        fig.savefig(_f)
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
