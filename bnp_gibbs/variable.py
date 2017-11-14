"""
"""
from collections.abc import MutableSequence

class Variable(MutableSequence):
    """ state-space variable with sample trace storage """
    def __init__(self, init=None, *args, **kwargs):
        MutableSequence.__init__(self, *args, **kwargs)
        self._trace = []
        if init:
            self._trace.append(init)

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

    @property
    def value(self):
        """get the most recently inserted value (current value)"""
        return self.__getitem__(-1)

    @value.setter
    def value(self, v):
        self.__setitem__(-1, v)

