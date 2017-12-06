#!/usr/bin/env python3
import sys
sys.path.insert(0, './bnp_tumorseg')
import logging
from bnp_tumorseg import hdpcluster

# setup logger
logger = logging.getLogger()


if __name__ == '__main__':
    hdpcluster.execute(root='.')
