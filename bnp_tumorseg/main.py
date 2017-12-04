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
    parser.add_argument('--resume-from', type=str, default=None, help='continue sampling from pickled results path')
    # parse args
    args = parser.parse_args()
    ftype = args.ftype
    dataset = args.dataset
    datapath = os.path.join(data_root, dataset)
    smoothlvl = max(0, args.smoothlvl)
    resamplefactor = max(0.01, min(4, args.resamplefactor))
    visualize = args.visualize
    NOTIFY = args.notify
    verbose = args.verbose
    maxiter = args.maxiter
    burnin = args.burnin
    resume = args.resume_from

    if resume:
        with open(resume, 'rb') as f:
            dataset, ss_iter, hist_numclasses, hist_numclasses_active,\
                    docs, sizes, dim, t_coll, k_coll, evidence, m, n = pickle.load(f)
        ss_iter -= 1
        for trace in t_coll + k_coll:
            trace.burnin = burnin
        logger.info('resuming at iter {} from "{}" data in "{}"'.format(ss_iter, dataset, resume))

    # make output directories
    p_figs  = os.path.join(figs_dir, dataset)
    p_blobs = os.path.join(blobs_dir, dataset)
    p_logs  = os.path.join(logs_dir, dataset)
    for dname in [data_root, p_figs, p_logs, p_blobs]:
        os.makedirs(dname, exist_ok=True)

    # setup logger
    logger = loggers.RotatingFile(os.path.join(p_logs, 'main.log'), level=loggers.INFO)
    # reset logging level
    if verbose <= 0:
        logger.setLevel(loggers.INFO)
    else:
        logger.setLevel(loggers.DEBUG+1-verbose)

    # semi-permanent settings
    rand.seed(21) #numpy
    random.seed(21) #python
    init_nclasses = 3 # randomly init items into # groups per image and # global groups

    #==================#
    # Model Definition #
    #==================#
    # load data - load each image as a separate document (j)
    if not resume:
        logger.info('loading images from {}.....'.format(datapath))
        docs, sizes, dim = fileio.loadImageSet(datapath, ftype=ftype, resize=resamplefactor)
    Nj = len(docs)                       # number of images
    Ni = [doc.shape[0] for doc in docs]  # list of image sizes (linear)
    if visualize:
        imcollection = [np.array(docs[j]).reshape((*sizes[j], dim))
                       for j in range(Nj)]
        fname = os.path.join(p_figs, '0_images')
        fileio.savefigure(imcollection, fname, cmap=None, header='input images', footer='resample factor: {}'.format(resamplefactor))
    logger.info('found {} images with dim={}'.format(len(docs), dim))

    # hyperparameter settings
    hp_gamma  = 1                # global DP concentration param (higher encourages more global classes to be created)
    hp_a0     = hp_gamma         # document-wise DP concentration param (higher encourages more document groups to be created)
    hp_n      = dim*2                # Wishart Deg. of Freedom (must be > d-1)
    hp_k      = 2                 # mean prior - covariance scaling param
    hp_mu     = np.ones((dim,))      # mean prior - location param (d-rank vector)
    hp_lbdinv = hp_n * 2*np.eye(dim) # mean prior - covariance matrix (dxd-rank matrix)
    # MRF params
    mrf_lbd   = smoothlvl            # strength of spatial group label smoothness

    # validate hyperparam settings
    assert hp_n >= dim
    assert hp_gamma > 0
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
    if not resume:
        evidence = [helpers.ModelEvidence() for i in range(init_nclasses)] # len==Nk at all times
        n = [[0]*init_nclasses for j in range(Nj)]  # len==Nj at all times for outerlist, len==Nt[j] at all times for inner list
                                          #     counts number of data items in doc j assigned to group t
                                          #     index as: n[j][t]
        m = [init_nclasses for i in range(init_nclasses)]   # len==Nk at all times; counts number of groups (t) with cluster assigned to k
                                          #     index as: m[k]
                                          # we can obtain m_dotdot (global number of groups) by summing elements in m

        # initialize latent parameters - traces will be saved
        #  z_coll = [Trace() for i in range(Nj)]    # nested collection of cluster assignment (int) traces for each item in each doc
        #                                           #     each is a numpy int array indicating full document cluster assignments
        #                                           #     index as: z_coll[j][i] - produces array of class assignment
        #  m_coll = [Trace()]                       # expected number of "groups" - len==Nk at all times, each Trace
        #                                           #     is array with shape=(Nj,)
        #                                           #     index as: m_coll[k][j]

        # nested collection of group assignment (int) traces for each item in each doc
        #   each item is np.array of integers between 0..(Tj)-1
        #   index as: t_coll[j].value[i]  [size doesnt change]
        t_coll = [Trace(burnin=burnin) for i in range(Nj)]

        # nested collection of cluster assignment (int) traces for each group in each
        #   doc. Each item is list of integers between 0..K-1
        #   index as: k_coll[j].value[t]  [size of inner list will change with Nt]
        k_coll = [Trace(burnin=burnin) for i in range(Nj)]

        # Properly initialize - random init among p groups per image and p global groups (p==init_nclasses)
        logger.debug("started adding all data items to initial class")
        for j, doc in enumerate(docs):
            t_coll[j].append( np.zeros((Ni[j],), dtype=np.uint32) )
            k_coll[j].append( [0]*init_nclasses )
            for t in range(init_nclasses):
                k_coll[j].value[t] = t
            for i, data in enumerate(doc):
                r = random.randrange(init_nclasses)
                evidence[r].insert(data)
                n[j][r] += 1
                t_coll[j].value[i] = r
        logger.debug("finished adding all data items to initial class")

        # history tracking variables
        hist_numclasses        = []
        hist_numclasses_active = []
        ss_iter = 0 # make available to function closures

    #==========#
    # Fxn Defs #
    #==========#
    def isClassEmpty(k):
        return m[k] <= 0
    def isGroupEmpty(j, t):
        return n[j][t] <= 0
    def numActiveGroups(j):
        return np.count_nonzero(n[j])
    def numActiveClasses():
        return np.count_nonzero(m)
    def numGroups(j):
        return len(k_coll[j].value)
    def numClasses():
        return len(m)
    def createNewGroup():
        pass
    def createNewClass():
        m.append(1)
        # create a tracked evidence object for the new class
        evidence.append(helpers.ModelEvidence())
        logger.debug2('new class created: k[{}]; {} active classes (+{} empty)'.format(
            numActiveClasses(), numActiveClasses(), numClasses()-numActiveClasses()))

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

    def make_class_maps():
        tcollection = [np.array(t_coll[j].mode(burn=(ss_iter>burnin))).reshape(sizes[j])
                       for j in range(Nj)]
        fname = os.path.join(p_figs, 'iter_{:04}_t'.format(ss_iter+1))
        fileio.savefigure(tcollection, fname, header='region labels', footer='iter: {}'.format(ss_iter+1))

        kcollection = [helpers.constructfullKMap(tcollection[j], k_coll[j].mode(burn=(ss_iter>burnin)))
                       for j in range(Nj)]
        fname = os.path.join(p_figs, 'iter_{:04}_k'.format(ss_iter+1))
        fileio.savefigure(kcollection, fname, header='class labels', footer='iter: {:4g}, # active classes: {}'.format(
            ss_iter+1, numActiveClasses()))

        # DEBUG
        if verbose >= 3:
            tcollection = [np.array(t_coll[j].value).reshape(sizes[j])
                           for j in range(Nj)]
            fname = os.path.join(p_figs, 'iter_{:04}_t_value'.format(ss_iter+1))
            fileio.savefigure(tcollection, fname, header='region labels', footer='iter: {}'.format(ss_iter+1))

            kcollection = [helpers.constructfullKMap(tcollection[j], k_coll[j].value)
                           for j in range(Nj)]
            fname = os.path.join(p_figs, 'iter_{:04}_k_value'.format(ss_iter+1))
            fileio.savefigure(kcollection, fname, header='class labels', footer='iter: {:4g}, # active classes: {}'.format(
                ss_iter+1, numActiveClasses()))

    def cleanup(fname="final_data.pickle"):
        """report final groups and classes"""
        logger.info('Sampling Completed')
        logger.info('Sampling Summary:\n'
                    '# active classes:               {:4g} (+{} empty)\n'.format(
                        numActiveClasses(), numClasses()+1-numActiveClasses() ) +
                    '# active groups (avg. per-doc): {:4g} (+{} empty)'.format(
                        np.average([numActiveGroups(j) for j in range(Nj)]),
                        np.average([numGroups(j)+1-numActiveGroups(j) for j in range(Nj)]) )
                    )

        # save tracked history plots
        savePlots()

        # save data to file
        fname = os.path.join(p_blobs, fname)
        with open(fname, 'wb') as f:
            pickle.dump([dataset, ss_iter, hist_numclasses, hist_numclasses_active,\
                         docs, sizes, dim, t_coll, k_coll, evidence, m, n], f)
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
    for _ in range(ss_iter, maxiter):
        ss_iter += 1
        logger.debug('Beginning Sampling Iteration {}'.format(ss_iter+1))

        # generate random permutation over document indices and iterate
        jpermutation = rand.permutation(Nj)
        for j in jpermutation:
            # create new trace histories from previous history
            t_coll[j].beginNewSample()
            k_coll[j].beginNewSample()

            # gen. rand. permutation over elements in document
            ipermutation = rand.permutation(Ni[j])
            for i in ipermutation:
                logger.debug3('ss_iter={}, j={}, i={}'.format(ss_iter, j, i))
                data = docs[j][i,:]

                m_items = [0]*len(m)
                for jj in range(Nj):
                    for tt in range(len(n[jj])):
                        m_items[k_coll[jj].value[tt]] += n[jj][tt]
                if True or [e.count for e in evidence] != m_items:
                    logger.debug3("Evidence counts and m, n containers do not agree\n" +
                                         "data in evidence: {}\n".format([e.count for e in evidence]) +
                                         "data in m: {}\n".format(m_items) +
                                         "m: {}\n".format(m) +
                                         "n[j]: {}\n".format(n[j]) +
                                         "k_coll[j].value: {}".format(k_coll[j].value) )

                # get previous assignments
                tprev = t_coll[j].value[i]
                kprev = k_coll[j].value[tprev]

                # remove count from group tprev, class kprev
                n[j][tprev] -= 1
                evidence[kprev].remove(data)
                logger.debug3('n[{}][{}]-- -> {}'.format(j, tprev, n[j][tprev]))
                # handle empty group in doc j
                if isGroupEmpty(j, tprev):
                    logger.debug2('Group {} in doc {} emptied'.format(tprev, j))
                    n[j][tprev] = 0 # probably not necessary
                    m[kprev] -= 1

                    # handle empty global cluster
                    if isClassEmpty(kprev):
                        m[kprev] = 0
                        logger.debug2('Class {} emptied'.format(kprev))

                # SAMPLING
                # sample tnext
                Nt = numGroups(j)
                Nk = numClasses()
                margL = np.zeros((Nk,))
                for kk in range(Nk):
                    if isClassEmpty(kk): continue
                    margL[kk] = helpers.marginalLikelihood(data, evidence[kk])
                margL_prior = helpers.marginalLikelihood(data, prior)
                mrf_args = (i, t_coll[j].value, sizes[j], mrf_lbd, k_coll[j].value)
                tnext = helpers.sampleT(n[j], k_coll[j].value, m+[hp_gamma], hp_a0, margL, margL_prior, mrf_args)
                t_coll[j].value[i] = tnext
                logger.debug3('tnext={} of [0..{}] (Nt={}, {} empty)'.format( tnext, Nt-1, Nt, Nt-numActiveGroups(j) ))
                if tnext >= Nt:
                    # conditionally sample knext for tnext=tnew
                    n[j].append(1)
                    logger.debug2('new group created: t[{}][{}]; {} active groups in doc {} (+{} empty)'.format(
                        j, tnext, numActiveGroups(j), j, Nt+1-numActiveGroups(j) ))
                    knext = helpers.sampleK(m+[hp_gamma], margL, margL_prior)
                    k_coll[j].value.append(knext)
                    logger.debug3('knext={} of [0..{}] ({} empty)'.format(knext, Nk-1, Nk-numActiveClasses()))
                    if knext >= Nk: createNewClass()
                    else: m[knext] += 1
                    logger.debug3('m[{}]++ -> {}'.format(knext, m[knext]))
                else:
                    n[j][tnext] += 1
                    knext = k_coll[j].value[tnext]
                    logger.debug3('n[{}][{}]++ -> {}'.format(j, tnext, n[j][tnext]))
                evidence[knext].insert(data)
                logger.debug3('')
            # END Pixel loop

            tpermutation = rand.permutation(numGroups(j))
            for t in tpermutation:
                if isGroupEmpty(j, t): continue

                # sampling from k dist where all data items from group tnext have been removed from a
                #     temporary model evidence object. Uses IID assumption and giving joint dist. as
                #     product of individual data item margLikelihoods
                Nk=numClasses()
                kprev = k_coll[j].value[t]
                m[kprev] -= 1

                # remove all data items from evidence of k_t
                evidence_copy = evidence[kprev].copy()
                data_t = docs[j][t_coll[j].value==t, :]
                for data in data_t:
                    evidence_copy.remove(data)

                # compute joint marginal likelihoods for data in group tnext
                jointMargL = np.zeros((Nk,))
                for kk in range(Nk):
                    if isClassEmpty(kk): continue
                    jointMargL[kk] = helpers.jointMarginalLikelihood(data_t, evidence_copy if kk==kprev else evidence[kk])
                jointMargL_prior = helpers.jointMarginalLikelihood(data_t, prior)
                knext = helpers.sampleK(m+[hp_gamma], jointMargL, jointMargL_prior)
                if knext >= Nk: createNewClass()
                else: m[knext] += 1

                # we can use reduced evidence as new evidence for kprev and add data to evidence for knext
                # if knext=kprev, we just leave the unmodified evidence object in place and do nothing
                if knext != kprev:
                    k_coll[j].value[t] = knext
                    evidence[kprev] = evidence_copy
                    for data in data_t:
                        evidence[knext].insert(data)
            #  time.sleep(10)
            # END group loop
        # END Image loop

        # save tracked history variables
        hist_numclasses_active.append(numActiveClasses())
        hist_numclasses.append(numClasses())

        # write current results
        if visualize:
            make_class_maps()

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
