"""
"""
from collections.abc import MutableSequence
import copy
import numpy as np
import scipy.stats

class Variable(MutableSequence):
    """state-space variable with sample trace storage and burn-in period rejection support"""
    def __init__(self, init=None, burnin=0, *args, **kwargs):
        MutableSequence.__init__(self, *args, **kwargs)
        self._trace = []
        if init:
            self._trace.append(init)

        # burn-in period - ignored by property getters
        self.burnin = burnin

    def __getitem__(self, i):
        return self._trace[i]

    def __setitem__(self, i, val):
        self._trace[i] = val

    def __delitem__(self, i):
        del self._trace[i]

    def __len__(self):
        return len(self._trace)

    def insert(self, i, v):
        self._trace.insert(i, v)

    def rollover(self):
        """Create new entry in the trace history which is identical to the previous trace"""
        prev = self._trace[-1]
        if isinstance(prev, np.ndarray):
            self._trace.append(prev.copy())
        else:
            self._trace.append(copy.deepcopy(prev))

    @property
    def value(self):
        """get the most recently inserted value (current value) from the trace history"""
        return self.__getitem__(-1)

    @value.setter
    def value(self, v):
        self.__setitem__(-1, v)

    @property
    def stable_samples(self):
        """Returns the portion of the trace history after the burn-in period"""
        return self._trace[self.burnin:]

    def mode(self, burn=True):
        """Provides a dimensionality-independent way of calculating the statistical mode with support for
        sampled arrays in the trace that have changed size over time. Non-existent values in smaller arrays
        are ignored in the mode calculation.

        Args:
            burn (bool): only values sampled after burn-in are considered
        """
        if burn:
            arr = self.stable_samples
        else:
            arr = self._trace

        target_size = np.array(arr[-1]).shape
        resized_arr = []
        for a in arr:
            a = np.array(a, dtype=np.float32)
            orig_size = a.shape
            res = np.resize(a, target_size)
            sliceobj = []
            for i in range(len(orig_size)):
                sliceobj.append(slice(orig_size[i], None))
            res[sliceobj] = np.NaN
            resized_arr.append(res)

        cat = np.stack(resized_arr, axis=np.array(arr[-1]).ndim)
        v = scipy.stats.mode(cat, axis=cat.ndim-1, nan_policy='omit')[0].astype(int)
        return v
