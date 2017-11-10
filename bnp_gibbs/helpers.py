from math import pi, sqrt
import numpy as np
import numpy.random as rand
from numpy.linalg import cholesky
from scipy.special import gammaln

def unStirling1st(n, m):
    """Computes unsigned Stirling numbers of the 1st kind by recursion
    see: https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind

    Args:
        n (int): # of elements
        m (int): # of disjoint cycles

    Returns:
        float: Number of permutations of n elements having m disjoint cycles
    """
    raise NotImplementedError()

def gammaln(x):
    """computes natural log of the gamma function which is more numerically stable than the gamma

    Args:
        x (float): any real value
    """
    return gammaln(x)


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

def marginalLikelihood(x, n_0, k_0, mu_0, ):
    """Computes marginal likelihood for a data vector given the model evidence of a Gaussian-Wishart
    conjugate prior model using cholesky decmomposition of scaling matrix to increase efficiency

    Args:
        x (np.ndarray): data vector [shape=(d,)]
        n_0 (float): wishart prior param
        k_0 (float): mean prior param - scale factor
        mu_0 (np.ndarray): mean prior param - location
        lbd_0 (np.ndarray): prior param - mean/scale


    """

    eps = 1e-9  # prevent divide-by-zero errors
    d   = 3
    sig = np.zeros((3,3))

    # intermediate steps
    nu = n_m - d + 1
    mu = mu_m
    b  = x - mu

    # compute student's T density given updated params
    tdensln = gammaln((nu+d)/2) / ( gammaln(nu/2)*(nu*pi)**(d/2) * sqrt(choleskyLogDet(L)) + eps ) \
            * (1 + (1/nu)*choleskyQuadForm(L,b) )**(-(nu+d)/2)

    return tdensln


