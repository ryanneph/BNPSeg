from math import pi, sqrt
import numpy as np
import numpy.ma as ma
from numpy.linalg import cholesky, solve
from copy import copy, deepcopy
import choldate
import logging
from . import loggers
from .wrappers import exp, log, gammaln

logger = logging.getLogger(__name__)

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


class ModelEvidenceNIW():
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
        obj = self.__class__()
        obj._dim = copy(self._dim)
        obj._count = copy(self._count)
        obj._sum = copy(self._sum)
        obj._outprod = copy(self._outprod)
        obj._cholcov = copy(self._cholcov)
        return obj

    def __deepcopy__(self, memodict):
        obj = self.__class__()
        obj._dim = deepcopy(self._dim, memodict)
        obj._count = deepcopy(self._count, memodict)
        obj._sum = deepcopy(self._sum, memodict)
        obj._outprod = deepcopy(self._outprod, memodict)
        obj._cholcov = deepcopy(self._cholcov, memodict)
        return obj

    def copy(self):
        return deepcopy(self)

    def insert(self, x):
        if ma.getmask(x).any():
            raise ValueError('attempt to insert masked data from evidence')
        self._count += 1
        self._insert_sum(x)
        #  self._insert_outprod(x)
        self._insert_cholcov(x)

    def remove(self, x):
        if ma.getmask(x).any():
            raise ValueError('attempt to remove masked data from evidence')
        if self._count <=0:
            raise RuntimeError('{} has no data items to remove'.format(self.__class__.__name__))
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

    def logMarginalLikelihood(self, x):
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
        d    = self.dim
        n_m  = self.n_m
        k_m  = self.k_m
        mu_m = self.mu_m
        L    = self.cholcov_m

        # intermediate steps - T-dist inputs
        nu = n_m - d + 1
        c = (k_m+1)/(k_m*nu)

        # compute student's T density given updated params
        tdensln = gammaln(0.5*(n_m+1)) - gammaln(0.5*nu) \
                - 0.5*(d*(log(nu*pi) + log(c)) + choleskyLogDet(L)) \
                - (0.5*(n_m+1))*log(1+ (1/(c*nu))*choleskyQuadForm(L, x-mu_m))
        if __debug__ and not tdensln<=0 and logger.getEffectiveLevel() <= loggers.DEBUG3:
            tdens = exp(tdensln)
            msg = "tdensln: {}\n".format(tdensln) + \
            "term 1: {}\n".format(gammaln(0.5*(n_m+1)) - gammaln(0.5*nu) - (0.5*d)*(log(nu*pi) + log(c))) + \
            "term 2: {}\n".format(choleskyLogDet(L)) + \
            "term 3: {}\n".format((0.5*(n_m+1))*log(1+ (1/(c*nu))*choleskyQuadForm(L, np.abs(x-mu_m)))) + \
            "tdens:  {}".format(tdens)
            logger.warning('result of marginal likelihood is not a valid probability between 0->1: {:0.3e}\n{}'.format(tdens, msg))
        return tdensln

    def marginalLikelihood(self, x):
        return exp(self.logMarginalLikelihood(x))

    def jointLogMarginalLikelihood(self, dataset):
        """Compute joint marginal likelihood for a set of data items assuming IID"""
        accum = 0
        for x in dataset:
            accum += self.logMarginalLikelihood(x)
        return accum

    def jointMarginalLikelihood(self, dataset):
        return exp(self.jointLogMarginalLikelihood(dataset))
