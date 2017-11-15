from math import pi, sqrt, log, exp
import numpy as np
import numpy.random as rand
from numpy.linalg import cholesky
from scipy import special
from datetime import datetime
import choldate


class ModelEvidence:
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

    def __init__(self, dim=1, **kwargs):
        # store evidence components
        self._dim = dim
        if 'count' in kwargs:
            self._count = kwargs['count']
        else:
            self._count = 0
        if 'sum' in kwargs:
            self._sum = kwargs['sum']
            #  print(f'set sum to {self._sum}')
        else:
            self._sum = np.zeros((dim,))
        if 'outprod' in kwargs:
            self._outprod = kwargs['outprod']
            #  print(f'set outprod to {self._outprod}')
        else:
            self._outprod = np.zeros((dim, dim))
        if 'Lstar' in kwargs:
            self._cholcov = kwargs['Lstar']
        elif 'covariance' in kwargs:
            self._cholcov = cholesky(kwargs['covariance']).T
        else:
            self._cholcov = np.zeros((dim, dim))

    def insert(self, x):
        self._count += 1
        self._insert_sum(x)
        self._insert_outprod(x)
        diff = self.mu_m - x
        self._insert_cholcov( sqrt((self.k_0 + self._count + 1)/(self.k_0 + self._count)) * diff )

    def remove(self, x):
        self._count -= 1
        self._remove_sum(x)
        self._remove_outprod(x)
        add = self.mu_m + x
        self._remove_cholcov( sqrt((self.k_0 + self._count - 1)/(self.k_0 + self._count)) * add )

    def _insert_sum(self, x):
        self._sum += x

    def _remove_sum(self, x):
        self._sum -= x

    def _insert_outprod(self, x):
        self._outprod += np.outer(x, x)

    def _remove_outprod(self, x):
        self._outprod -= np.outer(x, x)

    def _insert_cholcov(self, v):
        choldate.cholupdate(self._cholcov, v)

    def _remove_cholcov(self, v):
        choldate.choldowndate(self._cholcov, v)

    @property
    def count(self):
        return self._count

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
        return self._cholcov.T

    @property
    def avg(self):
        return self._sum / self._count

    @property
    def sum(self):
        return self._sum

    @property
    def outprod(self):
        return self._outprod


class DataAvg:
    """Maintains running sum of data items and exposes insert/remove methods and value property for
    on-the-fly average re-evaluation
    """
    def __init__(self, dim=1):
        self._accumulator = np.zeros((dim,))
        self._count = 0

    def insert(self, v):
        self._count += 1
        self._accumulator += v

    def remove(self, v):
        self._count -= 1
        self._accumulator -= v

    @property
    def value(self):
        return self._accumulator / self._count

class DataOuterProduct:
    """Maintains running outer product of data items and exposes insert/remove methods and value property"""
    def __init__(self, dim=1):
        self._accumulator = np.zeros((dim,dim))
        self._count = 0

    def insert(self, v):
        self._count += 1
        self._accumulator += np.outer(v, v)

    def remove(self, v):
        self._count -= 1
        self._accumulator -= np.outer(v, v)

    @property
    def value(self):
        return self._accumulator

class DataCholeskyCovariance:
    """Maintains running Cholesky Decomposition of Covariance matrix that can be efficiently updated/downdated
    using rank-1 Cholesky updates/downdates. Exposes insert/remove methods that implement the update/downdate
    and a value property exposing the cholesky decomposed matrix
    """
    def __init__(self, Lstar=None, covariance=None, dim=1):
        """Init from covariance, upper triangular cholesky decomp (Lstar) or zeros
        Note: _accumulator is assumed to be the upper triangular cholesky decomp matrix
        """
        if Lstar:
            self._accumulator = Lstar
        elif covariance:
            self._accumulator = cholesky(covariance).T
        else:
            self._accumulator = np.zeros(dim)
        self._count = 0

    def insert(self, v):
        self._count += 1
        choldate.cholupdate(self._accumulator, v.copy())

    def remove(self, v):
        self._count -=1
        choldate.choldowndate(self._accumulator, v.copy())

    @property
    def value(self):
        """Return lower triangular cholesky decomp matrix"""
        return self._accumulator.T

class unStirling1stProvider:
    """implements on demand caching version of Unsigned Stirling Number (1st Kind) provider """

    def __init__(self, precache_max=None, signed=False, verbose=False):
        self._cache = {}
        self._signed = signed
        if isinstance(precache_max, tuple) and len(precache_max) == 2:
            self.fillCache(*precache_max, verbose)

    def fillCache(self, n_max, m_max, verbose):
        """fill with cached values for all integer n, m up to provided values"""
        tstart = datetime.now()
        for n in range(n_max+1):
            for m in range(m_max+1):
                self.get(n, m, verbose>1)
        if verbose:
            print(f'Stirling number provider pre-caching to (n={n}, m={m}) completed in'
                  f' {(datetime.now()-tstart).microseconds/1000} ms')


    def get(self, n, m, verbose=False):
        try:
            return self._cache[(n,m)]
        except:
            val = self._eval(n, m)
            if not self._signed:
                val = abs(val)
            self._cache[(n, m)] = val
            if verbose: print(f'added value to cache: ({n}, {m})={val}')
            return val

    def _eval(self, n, m):
        """Computes unsigned Stirling numbers of the 1st kind by recursion
        see: https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind

        Args:
            n (int): # of elements
            m (int): # of disjoint cycles

        Returns:
            float: Number of permutations of n elements having m disjoint cycles
        """
        n1, m1 = n, m
        if n<=0:              return 1
        elif m<=0:            return 0
        elif (n==0 and m==0): return -1
        elif n!=0 and n==m:   return 1
        elif n<m:             return 0
        else:
            temp1=self.get(n1-1,m1)
            temp1=m1*temp1
            return (m1*(self.get(n1-1,m1)))+self.get(n1-1,m1-1)

def gammaln(x):
    """computes natural log of the gamma function which is more numerically stable than the gamma

    Args:
        x (float): any real value
    """
    return special.gammaln(x)

def choleskyR1Update(L, x):
    """Compute rank 1 update of the Lower Triangular matrix resulting from the Cholesky decomposition
    of a Positive Definite matrix such that the result, L_up, is the cholesky decomp of A_up = A + x*x'

    Args:
        L (np.ndarray): Lower triangular matrix [shape=(d,d)]
        x (np.ndarray): update vector [shape=(d,)]

    Returns:
        np.ndarray of same shape as L
    """
    raise NotImplementedError()

def choleskyR1Downdate(L, x):
    """Compute rank 1 downdate of the Lower Triangular matrix resulting from the Cholesky decomposition
    of a Positive Definite matrix such that the result, L_dw, is the cholesky decomp of A_dw = A - x*x'

    Args:
        L (np.ndarray): Lower triangular matrix [shape=(d,d)]
        x (np.ndarray): downdate vector [shape=(d,)]

    Returns:
        np.ndarray of same shape as L
    """
    raise NotImplementedError()

def choleskyQuadForm(L, b):
    """Compute quadratic form: b' * A^(-1) * b  using the cholesky inverse to reduce the problem to solving Lx=b where L is the
    Lower triangular matrix resulting from the cholesky decomp of A

    Args:
        L (np.ndarray): Lower triangular matrix [shape=(d,d)]
        b (np.ndarray): vector [shape=(d,)]
    """
    raise NotImplementedError()

def choleskyLogDet(L):
    """Compute  ln(|A|) = 2*sum[i=1-->d]{ ln(L_[i,i]) } : twice the log sum of the diag. elements of L, the
    lower triangular matrix resulting from the cholesky decomp. of A

    Args:
        L (np.ndarray): Lower triangular matrix [shape=(d,d)]

    Return:
        float
    """
    raise NotImplementedError()

def sampleBern(p):
    """Samples from a bernoulli distribution parameterized by p (and q=1-p)

    Args:
        p (float): bernoulli parameter limited by 0 <= p <= 1

    Returns: float drawn from bern. dist.
    """
    return rand.binomial(1, p)

def sampleDir(alpha):
    """draw random sample from dirichlet distribution parameterized by a vector of +reals

    Args:
        alpha (np.ndarray): vector of positive reals

    Returns:
        np.ndarray with same size as input vector
    """
    return rand.dirichlet(alpha)

def sampleM(a0, bk, njk, m_cap=20):
    """produces sample from distribution over M using normalized log probabilities parameterizing a
    categorical dist."""
    raise DeprecationWarning()

    wts = np.empty((m_cap,))
    sum = 0
    for m in range(m_cap):
        wts[m] = gammaln(a0*bk) - gammaln(a0*bk+njk) + log(stirling.get(njk, m)+1e-9) + m*(a0+bk)
        sum += wts[-1]
    wts = np.array(wts) / sum
    print(wts, np.sum(wts))
    return rand.multinomial(1, wts)


def sampleStudentT(loc, scale, df):
    """samples from generalized student's T distribution that falls in the location-scale family of
    distributions with 'df' degrees of freedom. As df->inf, the T-dist approaches the normal dist.
    For smaller values of df, the T dist is like a normal dist. with heavier tails.

    Args:
        loc (np.ndarray):    location parameter vector [shape=(d,)]
        scale (np.ndarray):  scale parameter matrix [shape=(d,d)]
        df (np.ndarray):     degrees of freedom [shape=(d,)]
    Returns:
        np.ndarray with shape=(d,)
    """
    return loc + scale.dot( rand.standard_t(df) )

def logMarginalLikelihood(x, evidence):
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
    eps = 1e-9  # prevent divide-by-zero errors
    d   = x.shape[0]

    # intermediate steps - bayesian param update
    #    note here that m denotes number of evidence data items in each cluster k
    n_m  = evidence.n_m
    k_m  = evidence.k_m
    mu_m = evidence.mu_m

    # intermediate steps - T-dist inputs
    nu = n_m - d + 1
    mu = mu_m
    sig = inv( (k_m*(n_m-d+1))/(k_m+1) * lbd_m )
    b  = x - mu

    # compute student's T density given updated params
    tdensln = gammaln((nu+d)/2) / ( gammaln(nu/2)*(nu*pi)**(d/2) * sqrt(choleskyLogDet(L)) + eps ) \
            * (1 + (1/nu)*choleskyQuadForm(L,b) )**(-(nu+d)/2)

    return tdensln

def logLikelihoodTnew(m, gamma, logMargL, logMargL_prior):
    """calculate data likelihood given that data item belongs to new group

    Args:
        m (list): Nk length list containing number of groups assigned to cluster k in all docs
        gamma (float): global DP param - concentration parameter (+real)
        logMargL (list): list of precomputed marginal likelihoods for each cluster k
        logMargL_prior (float): prior for data item given membership in new cluster knew
    """
    # TODO: CHECK CORRECT
    val = 0
    denom = np.array(m).sum() + gamma
    for k in range(len(m)):
        val += (exp(logMargL[k]) * m[k] )
    val += gamma * exp(logMargL_prior)
    val = log(val) - log(np.array(m).sum() + gamma)
    return val

def sampleT(k_j, m, a0, gamma, logMargL, logMargL_prior):
    """Draw t from a DP over Nt existing groups in the doc and one new group

    Args:
        m (list): Nk length list containing number of groups assigned to cluster k in all docs
        gamma (float): global DP param - concentration parameter (+real)
        logMargL (list): list of precomputed marginal likelihoods for each cluster k
        logMargL_prior (float): prior for data item given membership in new cluster knew
    """
    # TODO: CHECK CORRECT
    Nt = len(k_j)
    wts = np.zeros((Nt+1,))
    for t in range(Nt):
        # t=texist
        k = k_j[t]
        wts[t] = log(n[j][t]) + logMargL[k] + helpers.logMRF()
    wts[Nt] = log(a0) + logLikelihoodTnew(m, gamma, logMargL, logMargL_prior) # t=tnew
    # normalize
    wts = wts / wts.sum()
    print(f'sampleT - normalized with sum = {wts.sum()}')
    # draw t
    tnext = rand.multinomial(1, wts)
    return tnext

def logMRF():
    # TODO: NEED TO IMPLEMENT
    return 0


# INSTANCE OBJECTS
stirling = unStirling1stProvider()
