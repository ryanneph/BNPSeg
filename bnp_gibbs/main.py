#!/usr/bin/env python3
import sys
import os.path
import argparse
import numpy as np
import numpy.linalg
import numpy.random
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from variable import Variable
import fileio
import loggers

data_root = './test_files/'

if __name__ == '__main__':
    # arg defaults
    default_maxiter = 20
    default_ftype = 'float32'
    default_verbose = 2

    parser = argparse.ArgumentParser(description='Gibbs sampler for jointly segmenting vector-valued image collections',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-v', action='count', default=default_verbose, help='increase verbosity by 1 for each flag')
    parser.add_argument('--maxiter', type=int, default=default_maxiter)
    parser.add_argument('--ftype', type=str, choices=['float32', 'float64'], default=default_ftype, help='set floating point bit-depth')
    # parse args
    args = parser.parse_args()
    ftype = args.ftype
    verbose = args.verbose
    maxiter = args.maxiter

    # setup logger
    log = loggers.RotatingFile('./logs/main.log', level=loggers.DEBUG)

    # standardize usage of float type according to '--ftype' arg
    def float(x):
        return np.dtype(ftype).type(x)

    np.random.seed(20)
    eps = float(1e-9)  # prevent divide-by-zero errors

    # load data - load each image as a separate group (j)

    # hyperparameter settings
    hp_n      = 0
    hp_k      = 0
    hp_mu     = np.zeros((d,))  # d-rank vector
    hp_lbdinv = np.zeros((d,d)) # dxd-rank matrix - explicit inverse of lambda precision matrix


    # initialize latent parameters - traces will be saved
    z = Variable()  # cluster assignments (int)
    b = Variable()  #


    # Bookkeeping

    # convenience

    # Sampling
    ss_iter = 0
    while ss_iter<maxiter:
        ss_iter+=1

        # iterate j-group permutation
        for j in range(Nj):

            # remove this item from class
            label_was = lp_l[1][d]
            this_wordcounts = spcounts[label_was, :].reshape((W,))
            #   remove word counts from class "label_was"
            wordcounts[label_was] -= this_wordcounts
            #   remove doc count
            doccounts[label_was] -= 1

            # sample new most-likely label
            if verbose >= 3: sys.stdout.write('iter: {:d}, doc: {:d}'.format(ss_iter, d))
            label_is = cond_label(this_wordcounts)
            lp_l[0][d] += label_is  # add to running expectation tally
            lp_l[1][d] = label_is   # assign to current state space vector

            # add document to conditional
            #   add word counts to class "label_is"
            wordcounts[label_is] += this_wordcounts
            #   add doc count
            doccounts[label_is] += 1

            # track performance
            iter_num_flip += int(label_is!=label_was)
            if verbose >= 3: sys.stdout.write('\n')
            sys.stdout.flush()

        # sample thetas
        #   augment class wordcounts with hyperparam pseudo-counts
        iter_diff_lp_th = []
        for i in range(num_class):
            t = wordcounts[i] + hp_th
            lp_th[i][1] = np.random.dirichlet(t)
            lp_th[i][0] += lp_th[i][1]
