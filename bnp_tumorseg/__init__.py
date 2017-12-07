import os
import sys
import warnings

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())

# enable all warnings within the lib
warnings.filterwarnings('default', module='bnp_gibbs.*', category=DeprecationWarning)
warnings.filterwarnings('default', module='bnp_gibbs.*', category=PendingDeprecationWarning)
