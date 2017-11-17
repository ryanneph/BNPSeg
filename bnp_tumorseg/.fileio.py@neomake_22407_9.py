import sys
import os
import numpy as np
from PIL import Image
from loggers import RotatingFile

logger = RotatingFile

def loadImageSet(dname):
    """load a set of rgb images conatained within a single directory's top level
    Each image is loaded using PIL as an rgb image then linearized into a matrix [shape=(N,D)] containing N
    pixels in row-major (zyx) order, and D-dimensional pixel appearance features

    Args:
        dname (str): path to image collection containing directory
    """
    ims = []
    for fname in [x for x in os.listdir(dname) if os.path.isfile(x)]:
        with open(fname, 'r') as f:
            im = Image.open(f, 'r')
    return


