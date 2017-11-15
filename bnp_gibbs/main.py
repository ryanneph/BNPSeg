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
from helpers import stirling

data_root = './test_files/'

if __name__ == '__main__':
    # arg defaults
    default_maxiter = 5
    default_ftype = 'float32'
    default_dataset = 'balloons_sub'
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
    docs, dim = fileio.loadImageSet(datapath, verbose>1, ftype=ftype)
    Nj = len(docs)                       # number of images
    Ni = [doc.shape[0] for doc in docs]  # list of image sizes (linear)
    totaldataitems = np.sum(Ni)
    if verbose: print(f'found {len(docs)} images with dim={dim}')

    # initialize caching provider of Stirling Numbers
    #  stirling.fillCache(1000, 40, verbose>1)

    # hyperparameter settings
    hp_a0     = 1
    hp_gamma  = 1
    hp_n      = 1
    hp_k      = 1
    hp_mu     = np.zeros((dim,))  # d-rank vector
    hp_lbdinv = np.eye(dim) # dxd-rank matrix - explicit inverse of lambda precision matrix

    # bookkeeping vars
    # rather than recompute class avgs/scatter-matrix on each sample, maintain per-class data outer-product
    # and sum (for re-evaluating class avg) and simply update for each member insert/remove
    # each of these classes exposes insert(v)/remove(v) methods and value property
    helpers.ModelEvidence.n_0 = hp_n
    helpers.ModelEvidence.k_0 = hp_k
    helpers.ModelEvidence.mu_0 = hp_mu
    prior    = helpers.ModelEvidence(dim=dim, covariance=hp_lbdinv)
    evidence = [helpers.ModelEvidence(dim=dim,
                                      count=totaldataitems,
                                      sum=np.sum( np.sum(doc, axis=0) for doc in docs ),
                                      outprod=np.sum( np.sum( np.outer(doc[i,:], doc[i,:]) for i in range(doc.shape[0]) ) for doc in docs ),
                                      covariance=hp_lbdinv)]  # len==Nk at all times
    n = [[Ni[j]] for j in range(Nj)]  # len==Nj at all times for outerlist, len==Nt[j] at all times for inner list
                                      #     counts number of data items in doc j assigned to group t
                                      #     index as: n[j][t]
    m = [Nj]                          # len==Nk at all times; counts number of groups (t) with cluster assigned to k
                                      #     index as: m[k]
                                      # we can obtain m_dotdot (global number of groups) by summing elements in m

    # initialize latent parameters - traces will be saved
    #  z_coll = [Variable()]*Nj     # nested collection of cluster assignment (int) traces for each item in each doc
    #                               #     each is a numpy int array indicating full document cluster assignments
    #                               #     index as: z_coll[j][i] - produces array of class assignment
    #  m_coll = [Variable()]        # expected number of "groups" - len==Nk at all times, each Variable
    #                               #     is array with shape=(Nj,)
    #                               #     index as: m_coll[k][j]
    t_coll = [Variable()]*Nj     # nested collection of group assignment (int) traces for each item in each doc
                                 #     each item is np.array of integers between 0..(Tj)-1
                                 #     index as: t_coll[j].value[i]  [size doesnt change]
    k_coll = [Variable()]*Nj     # nested collection of cluster assignment (int) traces for each group in each
                                 #     doc. Each item is list of integers between 0..K-1
                                 #     index as: k_coll[j].value[t]  [size of inner list will change with Nt]
    b = Variable()               # wts on cat. distribition over k+1 possible cluster ids from root DP
                                 #     index as b[k] for k=1...Nk+1 (last element is wt of new cluster)

    # Properly initialize - all data items in a single group for each doc
    for j in range(Nj):
        t_coll[j].append( np.zeros((Ni[j],), dtype=np.uint32) )
        k_coll[j].append( [0] )
    b.append( helpers.sampleDir([1, 1]) )  # begin with uninformative sampling


    # Sampling
    for ss_iter in range(maxiter):
        if verbose>1: print(f'ss_iter={ss_iter}')

        # generate random permutation over document indices and iterate
        for j in rand.permutation(Nj):
            if verbose>2: print(f'j={j}')
            doc = docs[j]

            # gen. rand. permutation over elements in document
            for i in rand.permutation(Ni[j]):
                if verbose>2: print(f'i={i}')
                data = doc[i,:]

                # get previous assignments
                tprev = t_coll[j].value[i]
                kprev = k_coll[j].value[tprev]
                evidence_kprev = evidence[kprev]

                # remove count from group tprev, class kprev
                n[j][tprev] -= 1
                # handle empty group in doc j
                if n[j][tprev] <= 0:
                    if verbose>1: print(f'Group {tprev} in doc {j} emptied')
                    del n[j][tprev]       # forget number of data items in empty group
                    del k_coll[j][tprev]  # forget cluster assignment for empty group
                    m[kprev] -= 1

                # handle empty global cluster
                if m[kprev] <= 0:
                    if verbose>1: print(f'Cluster {kprev} emptied')
                    del m[kprev]
                    del evidence[kprev]
                else:
                    # remove data item from evidence for class k
                    evidence_kprev.remove(data)

                # SAMPLING
                # sample tnext
                Nt = len(k_coll[j])
                Nk = len(m)
                margL = -1*np.ones((Nk,))
                for k in range(Nk):
                    margL[k] = helpers.logMarginalLikelihood(data, evidence[k])
                margL_prior = helpers.logMarginalLikelihood(data, prior)
                tnext = helpers.sampleT(k_coll[j].value, m, hp_a0, hp_gamma, margL, margL_prior)

                # conditionally sample knext if tnext=tnew

                # TODO: get previous assignments in sampling step
                knext = kprev

                # add count to group tnext, class knext
                n[j][tnext] += 1
                m[knext] += 1


                # insert data into newly assigned cluster evidence
                evidence_knext = evidence[knext]
                evidence_knext.insert(data)
