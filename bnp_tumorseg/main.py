#!/usr/bin/env python3
import sys
import signal
import os.path
import argparse
import logging
import time
import math
import pickle
import random
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
if not 'DISPLAY' in os.environ:
    # no display server
    plt.switch_backend('agg')

from trace import Trace
import fileio
import loggers
import helpers
from notifications import pushNotification

NOTIFY = False  # send push notifications on success/exception?
CLEANUP = None

# setup directory structure
data_root = './test_files/'
figs_dir  = './figures/'
logs_dir  = './logs/'
blobs_dir = './blobs/'

# setup logger
logger = logging.getLogger()

def run_sampler():
    global NOTIFY, CLEANUP, logger

    # arg defaults
    default_maxiter        = 30
    default_burnin         = 40
    default_smoothlvl      = 10
    default_resamplefactor = 0.25
    default_ftype          = 'float64'
    default_dataset        = 'blackwhite_sub'
    default_visualize      = True
    default_verbose        = 0

    parser = argparse.ArgumentParser(description='Gibbs sampler for jointly segmenting vector-valued image collections',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-v', action='count', default=default_verbose, help='increase verbosity level by 1 for each flag')
    parser.add_argument('--visualize', action='store_true', default=default_visualize, help='produce intermediate/final result figures')
    parser.add_argument('--notify', action='store_true', default=NOTIFY, help='send push notifications')
    parser.add_argument('--maxiter', type=int, default=default_maxiter, help='maximum sampling iterations')
    parser.add_argument('--burnin', type=int, default=default_burnin, help='number of initial samples to discard in prediction')
    parser.add_argument('--dataset', type=str, choices=[x for x in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, x))],
                        default=default_dataset, help='named testing dataset in {}'.format(data_root))
    parser.add_argument('--smoothlvl', type=np.float, default=default_smoothlvl, help='Set the level of smoothing on class labels')
    parser.add_argument('--resamplefactor', type=np.float, default=default_resamplefactor, help='Set the resampling factor applied to input images')
    parser.add_argument('--ftype', type=str, choices=['float32', 'float64'], default=default_ftype, help='set floating point bit-depth')
    # parse args
    args = parser.parse_args()
    ftype = args.ftype
    datapath = os.path.join(data_root, args.dataset)
    smoothlvl = max(0, args.smoothlvl)
    resamplefactor = max(0.01, min(4, args.resamplefactor))
    visualize = args.visualize
    NOTIFY = args.notify
    verbose = args.verbose
    maxiter = args.maxiter
    burnin = args.burnin

    # make output directories
    p_figs  = os.path.join(figs_dir, args.dataset)
    p_blobs = os.path.join(blobs_dir, args.dataset)
    p_logs  = os.path.join(logs_dir, args.dataset)
    for dname in [data_root, p_figs, p_logs, p_blobs]:
        os.makedirs(dname, exist_ok=True)

    # setup logger
    logger = loggers.RotatingFile(os.path.join(p_logs, 'main.log'), level=loggers.INFO)
    # reset logging level
    if verbose <= 0:
        logger.setLevel(loggers.INFO)
    else:
        logger.setLevel(loggers.DEBUG+1-verbose)

    # standardize usage of float type according to '--ftype' arg
    def float(x):
        return np.dtype(ftype).type(x)

    # semi-permanent settings
    rand.seed(21)
    init_nclasses = 2 # randomly init items into # groups per image and # global groups

    #==================#
    # Model Definition #
    #==================#
    # load data - load each image as a separate document (j)
    logger.info('loading images from {}.....'.format(datapath))
    docs, sizes, dim = fileio.loadImageSet(datapath, ftype=ftype, resize=resamplefactor)
    Nj = len(docs)                       # number of images
    Ni = [doc.shape[0] for doc in docs]  # list of image sizes (linear)
    totaldataitems = np.sum(Ni)
    if visualize:
        imcollection = [np.array(docs[j]).reshape((*sizes[j], dim))
                       for j in range(Nj)]
        fname = os.path.join(p_figs, '0_images')
        fileio.savefigure(imcollection, fname, cmap=None, header='input images', footer='resample factor: {}'.format(resamplefactor))
    logger.info('found {} images with dim={}'.format(len(docs), dim))

    # hyperparameter settings
    #  hp_gamma  = 5                # global DP concentration param (higher encourages more global classes to be created)
    hp_a0     = 5                 # document-wise DP concentration param (higher encourages more document groups to be created)
    hp_n      = dim*2                # Wishart Deg. of Freedom (must be > d-1)
    hp_k      = 2                 # mean prior - covariance scaling param
    hp_mu     = 0.1*np.ones((dim,))      # mean prior - location param (d-rank vector)
    hp_lbdinv = hp_n * 2*np.eye(dim) # mean prior - covariance matrix (dxd-rank matrix)
    # MRF params
    mrf_lbd   = smoothlvl            # strength of spatial group label smoothness

    # validate hyperparam settings
    assert hp_n >= dim
    assert hp_a0 > 0
    assert hp_k > 0
    assert hp_lbdinv.ndim == 2 and hp_lbdinv.shape[0] == hp_lbdinv.shape[1] == dim

    # bookkeeping vars
    # rather than recompute class avgs/scatter-matrix on each sample, maintain per-class data outer-product
    # and sum (for re-evaluating class avg) and simply update for each member insert/remove
    # each of these classes exposes insert(v)/remove(v) methods and value property
    helpers.ModelEvidence.dim_0 = dim
    helpers.ModelEvidence.n_0 = hp_n
    helpers.ModelEvidence.k_0 = hp_k
    helpers.ModelEvidence.mu_0 = hp_mu
    helpers.ModelEvidence.lbdinv_0 = hp_lbdinv
    prior    = helpers.ModelEvidence()
    evidence = [[helpers.ModelEvidence() for i in range(init_nclasses)] for j in range(Nj)] # len==Nk at all times
    n = [[0 for i in range(init_nclasses)] for j in range(Nj)]   # len==Nk at all times; counts number of groups (t) with cluster assigned to k
                                      #     index as: m[k]
                                      # we can obtain m_dotdot (global number of groups) by summing elements in m

    # initialize latent parameters - traces will be saved
    #  z_coll = [Trace() for i in range(Nj)]    # nested collection of cluster assignment (int) traces for each item in each doc
    #                                           #     each is a numpy int array indicating full document cluster assignments
    #                                           #     index as: z_coll[j][i] - produces array of class assignment
    #  m_coll = [Trace()]                       # expected number of "groups" - len==Nk at all times, each Trace
    #                                           #     is array with shape=(Nj,)
    #                                           #     index as: m_coll[k][j]


    # nested collection of cluster assignment (int) traces for each group in each
    #   doc. Each item is list of integers between 0..K-1
    #   index as: k_coll[j].value[t]  [size of inner list will change with Nt]
    k_coll = [Trace(burnin=burnin) for i in range(Nj)]


    # Properly initialize - random init among p groups per image and p global groups (p==init_nclasses)
    logger.debug("started adding all data items to initial class")
    for j, doc in enumerate(docs):
        k_coll[j].append( [0]*Ni[j] )
        for i, data in enumerate(doc):
            r = random.randrange(init_nclasses)
            evidence[j][r].insert(data)
            n[j][r] += 1
            k_coll[j].value[i] = r
    logger.debug("finished adding all data items to initial class")

    # history tracking variables
    ss_iter = None # make available to function closures
    hist_numclasses        = []
    hist_numclasses_active = []

    #==========#
    # Fxn Defs #
    #==========#
    def isClassEmpty(j,k):
        return n[j][k] <= 0
    def numActiveClasses(j):
        return np.count_nonzero(n[j])
    def numClasses(j):
        return len(n[j])
    def createNewGroup():
        pass
    def createNewClass(j):
        n[j].append(1)
        # create a tracked evidence object for the new class
        evidence[j].append(helpers.ModelEvidence())
        logger.debug2('new class created: k[{}]; {} active classes (+{} empty)'.format(
            numActiveClasses(j), numActiveClasses(j), numClasses(j)-numActiveClasses(j)))

    def savePlots():
        """Create iteration histories of useful variables"""
        axsize = (0.1, 0.1, 0.8, 0.8)
        figmap = {'numclasses': {'active': hist_numclasses_active, 'total': hist_numclasses}}

        fig = plt.figure()
        for title, axmap in figmap.items():
            fname = os.path.join(p_figs, '0_hist_{}.png'.format(title))
            ax = fig.add_axes(axsize)
            for label, data in axmap.items():
                if len(data):
                    ax.plot(range(1, len(data)+1), data, label=label)
                    ax.set_xlabel('iteration #')
            ax.legend()
            ax.set_title(title)
            fig.savefig(fname)
            fig.clear()
        plt.close(fig)

    def cleanup(fname="final_data.pickle"):
        """report final groups and classes"""
        logger.info('Sampling Completed')
        logger.info('Sampling Summary:\n'
                    '# active classes:               {:4g} (+{} empty)\n'.format(
                        numActiveClasses(j), numClasses(j)+1-numActiveClasses(j) )
                    )

        # save tracked history plots
        savePlots()

        # save data to file
        fname = os.path.join(p_blobs, fname)
        with open(fname, 'wb') as f:
            pickle.dump([k_coll, sizes], f)
            logger.info('data saved to "{}"'.format(fname))
    CLEANUP = cleanup

    # register SIGINT handler (ctrl-c)
    def exit_early(sig, frame):
        # kill immediately on next press of ctrl-c
        signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(1))

        logger.warning('SIGINT recieved. Cleaning up and exiting early')
        cleanup(fname='data@iter#{}.pickle'.format(ss_iter))
        sys.exit(1)
    signal.signal(signal.SIGINT, exit_early)

    #==========#
    # Sampling #
    #==========#
    ss_iter = 0
    for ss_iter in range(maxiter):
        logger.debug('Beginning Sampling Iteration {}'.format(ss_iter+1))

        # generate random permutation over document indices and iterate
        jpermutation = rand.permutation(Nj)
        for j in jpermutation:
            # create new trace histories from previous history
            k_coll[j].beginNewSample()

            # gen. rand. permutation over elements in document
            ipermutation = rand.permutation(Ni[j])
            for i in ipermutation:
                logger.debug3('ss_iter={}, j={}, i={}'.format(ss_iter, j, i))
                data = docs[j][i,:]

                # get previous assignments
                kprev = k_coll[j].value[i]

                # remove count from group tprev, class kprev
                n[j][kprev] -= 1
                evidence[j][kprev].remove(data)
                logger.debug3('n[{}][{}]-- -> {}'.format(j, i, n[j][kprev]))
                # handle empty group in doc j
                if isClassEmpty(j, kprev):
                    logger.debug2('Group {} in doc {} emptied'.format(i, j))
                    n[j][kprev] = 0 # probably not necessary

                # SAMPLING
                # sample tnext
                Nk = numClasses(j)
                margL = np.zeros((Nk,))
                for kk in range(Nk):
                    if isClassEmpty(j,kk): continue
                    margL[kk] = helpers.marginalLikelihood(data, evidence[j][kk])
                margL_prior = helpers.marginalLikelihood(data, prior)
                mrf_args = (i, k_coll[j].value, sizes[j], mrf_lbd, k_coll[j].value)
                knext = helpers.sampleT(n[j], k_coll[j].value, n, hp_a0, margL, margL_prior)
                print(j, i, knext, len(n[j]), len(k_coll[j].value))
                k_coll[j].value[i] = knext
                if knext >= Nk:
                    # conditionally sample knext for tnext=tnew
                    createNewClass(j)
                    k_coll[j].value[i] = (knext)
                    logger.debug2('new group created: t[{}][{}]; {} active groups in doc {} (+{} empty)'.format(
                        j, knext, numActiveClasses(j), j, Nk+1-numActiveClasses(j) ))
                else:
                    n[j][knext] += 1
                    #  knext = k_coll[j].value[tnext]
                    logger.debug3('n[{}][{}]++ -> {}'.format(j, i, n[j][knext]))
                evidence[j][knext].insert(data)
                logger.debug3('')
            # END Pixel loop
        # END Image loop

        # save tracked history variables
        hist_numclasses_active.append(numActiveClasses(j))
        hist_numclasses.append(numClasses(j))

        # display class maps
        if visualize:
            kcollection = [np.array(k_coll[j].mode(burn=(ss_iter>burnin))).reshape(sizes[j])
                           for j in range(Nj)]
            fname = os.path.join(p_figs, 'iter_{:04}_t'.format(ss_iter+1))
            fileio.savefigure(kcollection, fname, header='region labels', footer='iter: {}'.format(ss_iter+1))

            # DEBUG
            if verbose >= 3:
                kcollection = [np.array(k_coll[j].value).reshape(sizes[j])
                               for j in range(Nj)]
                fname = os.path.join(p_figs, 'iter_{:04}_t_value'.format(ss_iter+1))
                fileio.savefigure(kcollection, fname, header='region labels', footer='iter: {}'.format(ss_iter+1))

    # log summary, generate plots, save checkpoint data
    cleanup()

if __name__ == '__main__':
    try:
        run_sampler()
        if NOTIFY: pushNotification('Success - {}'.format(__name__), 'finished sampling')

    except Exception as e:
        msg = 'Exception occured: {!s}'.format(e)
        logger.exception(msg)
        if NOTIFY: pushNotification('Exception - {}'.format(__name__), msg)
        if callable(CLEANUP): CLEANUP()
