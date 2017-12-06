import math
import numpy as np
import numpy.random as rand
from scipy import special
import logging
from . import loggers
from .wrappers import exp, log, gammaln

logger = logging.getLogger(__name__)

def sampleCatDist(wts):
    """Draw 0-based index from categorical distribution"""
    # check for inf/nan/extremes
    wtsum = np.sum(wts)
    if wtsum<=0 or wtsum==np.NaN or wtsum==np.Inf:
        # usually occurs becuase margL is 0 for all classes
        # TODO: should set to [0, ..., 0, 1] indicating guaranteed new class
        logger.warning("unstable categorical weights vector with sum(wts)={}.".format(wtsum))
        #  raise RuntimeError
        wts = np.ones_like(wts) / len(wts)
        logger.warning("Setting uniform weights: {}".format(wts))
    else:
        # normalize
        wts = wts / np.sum(wts)
        logger.debug3('cat wts: {}'.format(wts))
        # check for invalid params
        if __debug__ and not ((0<=wts).all and (wts<=1).all() and math.isclose(np.sum(wts), 1)):
            raise AssertionError("invalid categorical weights vector: {}".format(str(wts)))
    return np.nonzero(rand.multinomial(1, wts))[0][0]

def sampleDirDist(alpha):
    """draw random sample from dirichlet distribution parameterized by a vector of +reals

    Args:
        alpha (np.ndarray): vector of positive reals

    Returns:
        np.ndarray with same size as input vector
    """
    return rand.dirichlet(alpha).tolist()

def likelihoodTnew(beta, margL, margL_prior):
    """calculate data likelihood given that data item belongs to new group tnew
    computes: P(x_ji|tvect, t_ji=tnew, kvect)

    Args:
        beta (list): Nk+1 length list containing DP sampled "number of groups" assigned to cluster k in all docs
                     where the last element is beta_u: the DP sampled document concentration parameter
        margL (list): list of precomputed marginal likelihoods for each cluster k
        margL_prior (float): prior for data item given membership in new cluster knew
    """
    val = 0
    for k in range(len(beta)-1):
        if beta[k] <= 0:
            continue
        val += margL[k] * beta[k]
    val += beta[-1] * margL_prior
    val /= np.sum(beta)
    if __debug__ and not 0<=val<=1 and logger.getEffectiveLevel() <= loggers.DEBUG3:
        logger.warning('invalid likelihood encountered: {}'.format(val))
    return val

def sampleT(n_j, k_j, beta, a0, margL, margL_prior, mrf_args=None):
    """Draw t from a DP over Nt existing groups in the doc and one new group

    Args:
        n_j (list):  Nt length list containing number of data items in group t in doc j
        k_j (list):  Nt length list containing index of class assigned to group t in doc j
        beta (list): Nk+1 length list containing DP sampled "number of groups" assigned to cluster k in all docs
                     where the last element is beta_u: the DP sampled document concentration parameter

        margL (list): list of precomputed marginal likelihoods for each cluster k
        margL_prior (float): prior for data item given membership in new cluster knew

    Optional Args:
        mrf_args (tuple): tuple of valid arguments to be passed to MRF() (if None, don't use MRF)
    """
    Nt = len(n_j)
    wts = np.zeros((Nt+1,))
    for t in range(Nt):
        if n_j[t] <= 0:
            continue
        wts[t] = n_j[t] * margL[k_j[t]] # t=texist
        if mrf_args is not None:
            wts[t] *= MRF(t, *mrf_args)
    wts[Nt] = a0 * likelihoodTnew(beta, margL, margL_prior) # t=tnew
    # draw t
    tnext = sampleCatDist(wts)
    return tnext

def sampleK(beta, margL, margL_prior, mrf_args=None):
    """draw k from Nk existing global classes and one new class
    specify mrf_args if MRF constraint should be used directly on k-map"""
    Nk = len(beta)-1
    wts = np.zeros((Nk+1,))
    for k in range(Nk):
        if beta[k] <= 0:
            continue
        # k=kexist
        try:
            wts[k] = beta[k] * margL[k]
        except:
            print(wts.shape, len(beta), len(margL))
        if mrf_args is not None:
            wts[k] *= MRF(k, *mrf_args)
    wts[Nk] = beta[-1] * margL_prior # k=knew
    # draw k
    knext = sampleCatDist(wts)
    return knext

def sampleBeta(m, hp_gamma):
    """Sample beta from dirichlet distribution, filling beta_k=0 where m_k==0"""
    m = np.array(m)
    alpha = np.append(m[m>0], hp_gamma)
    res = sampleDirDist(alpha)
    active_counter = 0
    beta = []
    for k in range(len(m)):
        if m[k]<=0:
            beta.append(0)
        else:
            beta.append(res[active_counter])
            active_counter += 1
    beta.append(res[-1])
    return beta

def augmentBeta(beta, gamma):
    b = rand.beta(1, gamma)
    bu = beta[-1]
    beta[-1] = b*bu
    beta.append((1-b)*bu)
    return beta

def MRF(v, i, t_j, imsize, lbd, k_j=None):
    """compute Markov Random Field constraint probability using group labels of neighboring data items

    Args:
        v (int): indep. variable class index. treat as t for k_j==None and k for k_j!=None
        t (int): independent var. group index for this probability calculation
        i (int): linear index of data item into doc (image) j
        t_j (np.ndarray): Ni[j] length vector of group assignments which represents linearized image
            in row-major order indexed by i
        imsize (2-tuple): original dimensions of document (image) used to eval linear indices of neighboring
            data items
        lbd (float): +real weighting factor influencing the strength of the MRF constraint during inference
        w_j (np.ndarary): Ni[j] sized square matrix of pairwise edge weights in Markov Graph
    Optional Args:
        k_j (np.ndarray): vector mapping group label: t[j][i] to class label: k[j][t]. If None, use MRF on
            t-label instead of k-label
    """
    val = 0
    xi = i % imsize[1]
    yi = math.floor(i / imsize[1])
    # accumulate edge costs in clique
    for xo in [-1, 0, 1]:
        for yo in [-1, 0, 1]:
            if xo==0 and yo==0: continue
            x = xi + xo
            y = yi + yo
            # boundary handling - fill 0s
            if not (0<=x<imsize[1]) or not (0<=y<imsize[0]): continue
            t_ji = t_j[y*imsize[1]+x]
            if k_j is None:
                val += int(v == t_ji)
            else:
                val += int(k_j[v] == k_j[t_ji])
    val = math.exp(lbd * val)
    return val

def constructfullKMap(tmap, kmap):
    """construct a complete k-map from the complete t-map and mapping between t-vals and k-vals"""
    newarr = tmap.copy()
    for t, k in enumerate(kmap):
        newarr[tmap==t] = k
    return newarr
