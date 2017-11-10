# Bayesian Non-Parametric Tumor Segmentation

This project implements a _Gibbs Sampler_ for performing intra-tumor clustering of vector-valued voxel representations in a joint _Hierarchical Dirichlet Process-Markov Random Field_ bayesian non-parametric model. 

The model places each voxel across a population of patients' images into one of a variable number of classes. Utilization of the __HDP__ guarantees correspondence of a class definition for all patients, allowing the algorithm to select the appropriate number of distinct voxel representations based on population information. 

The __MRF__ constraint is placed on the class assignment to reinforce the assumption that under a revealing representation (feature vector) voxels (collections of cells) tend to compose themselves into larger clusters rather than spread them selves into small, disjoint clusters.

The algorithm draws samples from conditional distributions over the full state space vector to infer the latent class assignment parameters and implicit number of unique cell classifications.
