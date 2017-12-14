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
    """Wrapper for updating cluster evidence through insertion/removal operations of each data item"""

    def __init__(self, n, k, mu, cov, U=None):
        """initialize the prior"""
        mu = np.atleast_1d(mu)
        cov = np.atleast_2d(cov)
        self._count = 0
        self._dim   = mu.size
        self._n     = n
        self._k     = k
        self._mu    = mu
        self._U     = cholesky(cov + k*np.outer(mu, mu)).T
        if U is not None: self._U = np.atleast_2d(U)

        # caching
        self._cache = {}

        # validate hyperparam settings
        assert n >= self._dim
        assert k > 0
        assert cov.shape == tuple([self._dim]*2)

    def __str__(self):
        return '{}\n'.format(self.__repr__()) +\
               'count: {}\n'.format(self.count) +\
               'dim:   {}\n'.format(self.dim) +\
               'n:     {}\n'.format(self.n) +\
               'k:     {}\n'.format(self.k) +\
               'mu:    {}\n'.format(self.mu) +\
               '_U:    {}\n'.format(self._U) + \
               'U:     {}\n'.format(self.U)

    def _resetCache(self):
        """remove all cached vars"""
        self._cache = {}

    def _insertOne(self, x):
        assert x.ndim <= 1
        assert x.size == self._dim
        self._count += 1
        self._n  += 1
        self._k  += 1
        self._mu  = self._mu + (x-self._mu)/self._k
        choldate.cholupdate(self._U, np.copy(x))

    def _removeOne(self, x):
        assert x.ndim <= 1
        assert x.size == self._dim
        self._count -= 1
        self._n  -= 1
        self._k  -= 1
        self._mu  = self._mu - (x-self._mu)/self._k
        choldate.choldowndate(self._U, np.copy(x))

    def copy(self):
        new = deepcopy(self)
        assert self._U is not new._U
        return new

    def insert(self, x):
        x = np.atleast_1d(np.squeeze(x))
        if x.ndim <= 1:
            self._insertOne(x)
        elif x.ndim == 2:
            for item in x:
                self._insertOne(x)
        else:
            raise ValueError('input "{}" must have ndim == 1 or 2 not {}'.format(x, x.ndim))
        self._resetCache()

    def remove(self, x):
        if self._count <=0:
            raise RuntimeError('This {} has no data items to remove'.format(self.__class__.__name__))
        x = np.atleast_1d(np.squeeze(x))
        if x.ndim <= 1:
            self._removeOne(x)
        elif x.ndim == 2:
            for item in x:
                self._removeOne(x)
        else:
            raise ValueError('input "{}" must have ndim == 1 or 2 not {}'.format(x, x.ndim))
        self._resetCache()

    @property
    def count(self):
        return self._count

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, v):
        assert self._n >= v
        self._dim = v

    @property
    def nu(self):
        return (self._n - self._dim + 1)

    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def mu(self):
        return self._mu

    @property
    def U(self):
        if 'U' in self._cache:
            return self._cache['U']
        Utemp = np.copy((self._U))
        choldate.choldowndate(Utemp, sqrt(self._k)*self._mu)
        trueU = Utemp * sqrt( (1+(1/self._k))/(self.nu) )
        # update cache
        self._cache['U'] = trueU
        return trueU

    @property
    def L(self):
        return (self.U).T

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
        d  = self.dim
        nu = self.nu
        n  = self.n
        mu = self.mu
        L  = self.L

        # compute student's T density given updated params
        t1 = gammaln(0.5*(n+1)) - gammaln(0.5*nu)
        t2 = - 0.5*(d*log(nu*pi) + choleskyLogDet(L))
        t3 = - (0.5*(n+1))*log(1+(1/nu)*choleskyQuadForm(L, x-mu))
        tdensln = t1 + t2 + t3
        #TODO REMOVE DEBUG
        #  tdens = exp(tdensln)
        #  msg = "tdensln: {}\n".format(tdensln) + \
        #  "gammln1: {}\n".format(gammaln(0.5*(n+1))) + \
        #  "gammln2: {}\n".format(gammaln(0.5*(nu))) + \
        #  "chollogdet: {}\n".format(choleskyLogDet(L)) + \
        #  "L: {}\n".format(L) +\
        #  "diag(L): {}\n".format(np.diagonal(L)) +\
        #  "term 1: {}\n".format(t1+t2) + \
        #  "term 3: {}\n".format(t3) + \
        #  "tdens:  {}".format(tdens)
        #  print(msg)
        #TODO REMOVE END DEBUG
        if not tdensln<=0:
            tdens = exp(tdensln)
            msg = \
                "evidence:\n{}\n".format(self) + \
                "data:    {}\n".format(x) + \
                "term 1:  {}\n".format(t1) + \
                "term 2:  {}\n".format(t2) + \
                "term 3:  {}\n".format(t3) + \
                "tdensln: {}\n".format(tdensln) + \
                "tdens:   {}".format(tdens)
            raise ValueError('result of marginal likelihood is not a valid probability between 0->1: {:0.3e}\n{}'.format(tdens, msg))
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
