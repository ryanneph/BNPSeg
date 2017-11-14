#!/usr/bin/env python3
import sys
import os.path
import argparse
import numpy as np
import numpy.linalg
import numpy.random as rand
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from variable import Variable
import fileio
import loggers
import helpers

data_root = './test_files/'

if __name__ == '__main__':
    # arg defaults
    default_maxiter = 5
    default_ftype = 'float32'
    default_dataset = 'balloons'
    default_verbose = 2

    parser = argparse.ArgumentParser(description='Gibbs sampler for jointly segmenting vector-valued image collections',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-v', action='count', default=default_verbose, help='increase verbosity by 1 for each flag')
    parser.add_argument('--maxiter', type=int, default=default_maxiter)
    parser.add_argument('--dataset', type=str, choices=[x for x in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, x))],
                        default=default_dataset, help=f'named testing dataset in {data_root}')
    parser.add_argument('--ftype', type=str, choices=['float32', 'float64'], default=default_ftype, help='set floating point bit-depth')
    # parse args
    args = parser.parse_args()
    ftype = args.ftype
    datapath = os.path.join(data_root, args.dataset)
    verbose = args.verbose
    maxiter = args.maxiter

    # setup logger
    logger = loggers.RotatingFile('./logs/main.log', level=loggers.DEBUG)

    # standardize usage of float type according to '--ftype' arg
    def float(x):
        return np.dtype(ftype).type(x)

    # semi-permanent settings
    rand.seed(20)
    eps = float(1e-9)  # prevent divide-by-zero errors
    m_sample_cap = 40


    # load data - load each image as a separate group (j)
    if verbose: print(f'loading images from {datapath}.....', end=('\n' if verbose>1 else ''))
    docs, dim = fileio.loadImageSet(datapath, verbose>1)
    Nj = len(docs) # number of images
    if verbose: print(f'found {len(docs)} images with dim={dim}')

    # initialize caching provider of Stirling Numbers
    stirling = helpers.unStirling1stProvider((1000, 40), verbose>1)

    # hyperparameter settings
    hp_n      = 0
    hp_k      = 0
    hp_mu     = np.zeros((dim,))  # d-rank vector
    hp_lbdinv = np.zeros((dim,dim)) # dxd-rank matrix - explicit inverse of lambda precision matrix

    # initialize latent parameters - traces will be saved
    z_coll = [Variable()]*Nj     # nested collection of cluster assignment (int) traces for each item in each doc
                                 #     each is a numpy int array indicating full document cluster assignments
                                 #     index as: z_coll[j][i,:] - produces (d,)-shaped vector of item features
    m_coll = [Variable()]        # expected number of "groups" - len==Nk at all times, each Variable
                                 #     is array with shape=(Nj,)
                                 #     index as: m_coll[k][j]
    b = Variable()               # wts on cat. distribition over k+1 possible cluster ids from root DP
                                 #     index as b[k] for k=1...Nk+1 (last element is wt of new cluster)

    # bookkeeping vars
    # rather than recompute class avgs/scatter-matrix on each sample, maintain per-class data outer-product
    # and sum (for re-evaluating class avg) and simply update for each member insert/remove
    # each of these classes exposes insert(v)/remove(v) methods and value property
    avg = [helpers.DataAvg(dim)]           # len==Nk at all times
    op  = [helpers.DataOuterProduct(dim)]  # len==Nk at all times


    # Sampling
    ss_iter = 0
    while ss_iter<maxiter:
        ss_iter+=1

        # generate random permutation over document indices and iterate
        for j in rand.permutation(Nj):
            doc = docs[j]
            Ni = doc.shape[0]

            # gen. rand. permutation over elements in document
            for i in rand.permutation(Ni):
                # remove counts associated with this item from group t and class k
                kprev = z
                n_jk[(j,kprev)] -= 1

        #      # remove this item from class
        #      label_was = lp_l[1][d]
        #      this_wordcounts = spcounts[label_was, :].reshape((W,))
        #      #   remove word counts from class "label_was"
        #      wordcounts[label_was] -= this_wordcounts
        #      #   remove doc count
        #      doccounts[label_was] -= 1

        #      # sample new most-likely label
        #      if verbose >= 3: sys.stdout.write('iter: {:d}, doc: {:d}'.format(ss_iter, d))
        #      label_is = cond_label(this_wordcounts)
        #      lp_l[0][d] += label_is  # add to running expectation tally
        #      lp_l[1][d] = label_is   # assign to current state space vector

        #      # add document to conditional
        #      #   add word counts to class "label_is"
        #      wordcounts[label_is] += this_wordcounts
        #      #   add doc count
        #      doccounts[label_is] += 1

        #      # track performance
        #      iter_num_flip += int(label_is!=label_was)
        #      if verbose >= 3: sys.stdout.write('\n')
        #      sys.stdout.flush()

