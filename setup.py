from distutils.core import setup
from bnp_gibbs.version import VERSION_FULL

setup(
    name='bnp_tumorseg',
    version=VERSION_FULL,
    description='Hierarchical Bayesian Non-Parametric co-clustering of multi-channel natural and medical images',
    author='Ryan Neph',
    author_email="ryanneph@ucla.edu",
    packages=['bnp_tumorseg'],
)
