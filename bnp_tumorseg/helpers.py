import math
from math import pi, sqrt, exp
import numpy as np
import numpy.random as rand
from numpy.linalg import cholesky, solve
from scipy import special
from datetime import datetime
from copy import copy, deepcopy
import choldate
import logging
import loggers

logger = logging.getLogger(__name__)

eps = 1e-12
def log(x):
    return math.log(x+eps)

class ModelEvidence():
    """Wrapper for updating cluster evidence through insertion/removal operations of each data item
    The following properties are exposed and updated/downdated using the class insert/remove methods:

    avg            - keeps a running avg of data items in the cluster with efficient updating
    op             - keeps a running outer product of data items in cluster with efficient updating
    cholcov        - keeps a running upper triangular cholesky decomposition of the covariance with efficient
                     rank - 1 updates/downdates (internal storage is upper triangular Lstar)
    """
    # Class Static hyperparam storage
    n_0 = None
    k_0 = None
    mu_0 = None
    lbdinv_0 = None
    dim_0 = None

    def __init__(self, **kwargs):
        # store evidence components
        if 'dim' in kwargs:
            self._dim = kwargs['dim']
        elif self.dim_0 is not None:
            self._dim = self.dim_0
        else:
            self._dim = 1

        if 'count' in kwargs:
            self._count = kwargs['count']
        else:
            self._count = 0

        if 'sum' in kwargs:
            self._sum = kwargs['sum']
        else:
            self._sum = np.zeros((self._dim,))

        if 'outprod' in kwargs:
            self._outprod = kwargs['outprod']
        else:
            self._outprod = np.zeros((self._dim, self._dim))

        if 'Lstar' in kwargs:
            self._cholcov = kwargs['Lstar']
        elif 'covariance' in kwargs:
            self._cholcov = cholesky(kwargs['covariance']).T
        elif self.lbdinv_0 is not None:
            self._cholcov = cholesky(self.lbdinv_0)
        else:
            self._cholcov = np.zeros((self._dim, self._dim))

    def __copy__(self):
        obj = ModelEvidence()
        obj._dim = copy(self._dim)
        obj._count = copy(self._count)
        obj._sum = copy(self._sum)
        obj._outprod = copy(self._outprod)
        obj._cholcov = copy(self._cholcov)
        return obj

    def __deepcopy__(self, memodict):
        obj = ModelEvidence()
        obj._dim = deepcopy(self._dim, memodict)
        obj._count = deepcopy(self._count, memodict)
        obj._sum = deepcopy(self._sum, memodict)
        obj._outprod = deepcopy(self._outprod, memodict)
        obj._cholcov = deepcopy(self._cholcov, memodict)
        return obj

    def copy(self):
        return deepcopy(self)

    def insert(self, x):
        self._count += 1
        self._insert_sum(x)
        #  self._insert_outprod(x)
        self._insert_cholcov(x)

    def remove(self, x):
        if self._count <=0:
            raise RuntimeError('ModelEvidence has no data items to remove')
        self._count -= 1
        self._remove_sum(x)
        #  self._remove_outprod(x)
        self._remove_cholcov(x)

    def _insert_sum(self, x):
        self._sum += x

    def _remove_sum(self, x):
        self._sum -= x

    def _insert_outprod(self, x):
        self._outprod += np.outer(x, x)

    def _remove_outprod(self, x):
        self._outprod -= np.outer(x, x)

    def _insert_cholcov(self, v):
        # count has already been incremented by 1, so we subtract 1 from each count
        choldate.cholupdate(self._cholcov, sqrt((self.k_0 + self._count)/(self.k_0 + self._count-1)) * (self.mu_m - v) )

    def _remove_cholcov(self, v):
        # count has already been decremented by 1, so we add 1 from each count
        choldate.choldowndate(self._cholcov, sqrt((self.k_0 + self._count)/(self.k_0 + self._count+1)) * (self.mu_m - v) )

    @property
    def count(self):
        return self._count

    @property
    def dim(self):
        return self._dim

    # Model Parameter Updates
    @property
    def n_m(self):
        return self.n_0 + self.count

    @property
    def k_m(self):
        return self.k_0 + self.count

    @property
    def mu_m(self):
        return (self.k_0*self.mu_0 + self._sum) / (self.k_0 + self._count)

    @property
    def cholcov_m(self):
        # return lower triangular matrix L
        return self._cholcov.T
        #  return cholesky(self.lbdinv_0 + self.outprod - (self.k_0 + self.count)*np.outer(self.mu_m, self.mu_m) + self.k_0*np.outer(self.mu_0, self.mu_0)).T

    # Class Evidence States
    @property
    def sum(self):
        return self._sum

    @property
    def outprod(self):
        return self._outprod

    # Convenience
    @property
    def avg(self):
        if self._count <= 0: return 0
        else: return self._sum / self._count


def gamma(x):
    """wrapper for computes gamma function"""
    return special.gamma(x)

def gammaln(x):
    """computes natural log of the gamma function which is more numerically stable than the gamma"""
    return special.gammasgn(x)*special.gammaln(x)

def choleskyQuadForm(L, b):
    """Compute quadratic form: b' * A^(-1) * b  using the cholesky inverse to reduce the problem to solving Lx=b where L is the
    Lower triangular matrix resulting from the cholesky decomp of A

    Args:
        L (np.ndarray): Lower triangular matrix [shape=(d,d)]
        b (np.ndarray): vector [shape=(d,)]
    """
    fsubLinvb = solve(L, b)
    return fsubLinvb.dot(fsubLinvb)

def choleskyDet(L):
    """Compute  |A| = (product[i=1-->d]{ L_[i,i] })**2 : the square of the product of the diag. elements of L;
    with L being the lower triangular matrix resulting from the cholesky decomp. of A

    Args:
        L (np.ndarray): Lower triangular matrix [shape=(d,d)]
    """
    return np.product(np.diagonal(L))**2

def choleskyLogDet(L):
    """Compute  ln(|A|) = 2*sum[i=1-->d]{ ln(L_[i,i]) } : twice the log sum of the diag. elements of L, the
    lower triangular matrix resulting from the cholesky decomp. of A

    Args:
        L (np.ndarray): Lower triangular matrix [shape=(d,d)]
    """
    return 2*np.sum(np.log(np.diagonal(L)))

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

def logMarginalLikelihood(x: np.ndarray, evidence: ModelEvidence):
    """Computes marginal likelihood for a data vector given the model evidence of a Gaussian-Wishart
    conjugate prior model using cholesky decmomposition of scaling matrix to increase efficiency.

    All input arguments are assumed to be exclusive to cluster k already, thus the result is the marginal
    likelihood that data x is in cluster k. repeated execution with varying args is necessary to produce the
    full marg. likelihood over all clusters

    Args:
        x (np.ndarray): data vector [shape=(d,)]
        evidence (ModelEvidence): class containing updated bayesian params and running storage of various inputs

    Returns:
        float describing marginal likelihood that data x belongs to cluster k given model evidence avg
    """
    # intermediate steps - bayesian param update is already stored in evidence
    #    note here that m denotes number of evidence data items in each cluster k - not including x
    d    = evidence.dim
    n_m  = evidence.n_m
    k_m  = evidence.k_m
    mu_m = evidence.mu_m
    L    = evidence.cholcov_m

    # intermediate steps - T-dist inputs
    nu = n_m - d + 1
    c = (k_m+1)/(k_m*nu)

    # compute student's T density given updated params
    tdensln = gammaln(0.5*(n_m+1)) - gammaln(0.5*nu) \
            - 0.5*(d*(log(nu*pi) + log(c)) + choleskyLogDet(L)) \
            - (0.5*(n_m+1))*log(1+ (1/(c*nu))*choleskyQuadForm(L, x-mu_m))
    if __debug__ and not tdensln<=0 and logger.getEffectiveLevel() <= loggers.DEBUG3:
        tdens = exp(tdensln)
        msg = "tdensln: {}".format(tdensln) + \
        "term 1: {}".format(gammaln(0.5*(n_m+1)) - gammaln(0.5*nu) - (0.5*d)*(log(nu*pi) + log(c))) + \
        "term 2: {}".format(choleskyLogDet(L)) + \
        "term 3: {}".format((0.5*(n_m+1))*log(1+ (1/(c*nu))*choleskyQuadForm(L, np.abs(x-mu_m)))) + \
        "tdens:  {}".format(tdens)
        logger.warning('result of marginal likelihood is not a valid probability between 0->1: {:0.3e}\n{}'.format(tdens, msg))
    return tdensln

def marginalLikelihood(x, evidence):
    return exp(logMarginalLikelihood(x, evidence))

def jointLogMarginalLikelihood(dataset, evidence):
    """Compute joint marginal likelihood for a set of data items assuming IID"""
    accum = 0
    for x in dataset:
        accum += logMarginalLikelihood(x, evidence)
    return accum

def jointMarginalLikelihood(dataset, evidence):
    return exp(jointLogMarginalLikelihood(dataset, evidence))

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
            wts[k] *= MRF(*mrf_args, k=k)
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

def MRF(t, i, t_j, imsize, lbd, k_j=None, k=None):
    """compute Markov Random Field constraint probability using group labels of neighboring data items

    Args:
        i (int): linear index of data item into doc (image) j
        t (int): independent var. group index for this probability calculation
        t_j (np.ndarray): Ni[j] length vector of group assignments which represents linearized image
            in row-major order indexed by i
        imsize (2-tuple): original dimensions of document (image) used to eval linear indices of neighboring
            data items
        lbd (float): +real weighting factor influencing the strength of the MRF constraint during inference
        w_j (np.ndarary): Ni[j] sized square matrix of pairwise edge weights in Markov Graph
    Optional Args:
        k   (int): indep. variable class index. if None, use k=k_j[t] instead
        k_j (np.ndarray): vector mapping group label: t[j][i] to class label: k[j][t]. If None, use MRF on
            t-label instead of k-label
    """
    val = 0
    xi = i % imsize[1]
    yi = math.floor(i / imsize[1])
    # accumulate edge costs in clique
    for xo in [-1, 1]:
        for yo in [-1, 1]:
            x = xi + xo
            y = yi + yo
            # boundary handling - fill 0s
            if not (0<=x<imsize[1]) or not (0<=y<imsize[0]):
                continue
            t_ji = t_j[y*imsize[1]+x]
            if k_j is None:
                val += int(t == t_ji)
            else:
                if k is None:
                    k=k_j[t]
                val += int(k == k_j[t_ji])
    val = math.exp(lbd * val)
    return val

def constructfullKMap(tmap, kmap):
    """construct a complete k-map from the complete t-map and mapping between t-vals and k-vals"""
    newarr = tmap.copy()
    for t, k in enumerate(kmap):
        newarr[tmap==t] = k
    return newarr
