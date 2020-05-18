import numpy as np
from . import nndescent

def crs(objects: np.ndarray, coverage: float, similarity, K: int, threshold: float):
    """Select representatives via CRS method.

    :param objects: cluster samples to be represented
    :type objects: np.ndarray
    :param coverage: percentage of cluster to be covered by representatives
    :type coverage: float
    :param similarity: similarity function between two objects that returns 1 for similar objects and [0, 1) otherwise
    :type similarity: function
    :param K: order of the neighborhood in kNN graph
    :type K: int
    :param threshold: value to be used as threshold in creating reverse graph, cluster homogeneity recommended
    :type threshold: float
    :return: indices of objects selected as representatives from input objects
    :rtype: list
    """

    # values for sample rate are taken from NNDescent authors
    knn_graph = nndescent.NNDescentFull(dataset=objects,
                                        similarity=similarity,
                                        K=K,
                                        sample_rate=0.7,
                                        delta=0.001)

    representatives = nndescent.getReprIndicesReverseNeighborsThreshold(knn_graph, coverage, threshold)
    return representatives