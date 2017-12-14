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




# INSTANCE OBJECTS
stirling = unStirling1stProvider()
