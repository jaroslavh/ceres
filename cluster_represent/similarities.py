#file with all different similarities implemented for UPS experiments

import numpy as np
from math import sqrt #used in euclidSim
from numba import jit


@jit(nopython=True)
def euclid_sim(a, b):
    """Assumes a and b to be np.ndarray 1D arrays.
       Returns 1/1+d(a,b) where d is minkowski distance."""
    full_num = 0
    for i, j in enumerate(a):
        full_num += (j - b[i])**2
    return 1.0 / (1.0 + sqrt(full_num))


@jit(fastmath=True, cache=True)
# Edited from cosine distance in pynndescent
def cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 1.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    else:
        return result / np.sqrt(norm_x * norm_y)


# ======================================================================================================================

def freqSim(host_a, host_b, timewindow_num: int = 288):
    """Return frequency similarity between 2 hosts for given time.
    
    :param host_a: frequencies for each visited host {host1:freq1, host2:freq2}
    :type host_a: dict
    :param host_b: frequencies for each visited host {host1:freq1, host2:freq2}
    :type host_b: dict
    :param timewindow_num: number of 5 minute windows of given dataset
    :type timewindow_num: int
    :return: similarity between 2 hosts [0, 1]
    :rtype: float
    """
    ha = set(host_a.keys())
    hb = set(host_b.keys())
    S = list(ha.union(set(hb)))

    #no hosts in common
    if not ha.intersection(hb):
        return 0.0

    #calculating similarity
    sum_FaFb = 0.0 
    sum_Fa2 = 0.0 
    sum_Fb2 = 0.0
    for s in S:
        Fa = host_a.get(s, 0.0) / timewindow_num
        Fb = host_b.get(s, 0.0) / timewindow_num
        sum_FaFb += Fa * Fb 
        sum_Fa2 += Fa**2
        sum_Fb2 += Fb**2
    if sum_FaFb == 0.0:
        res = 0.0 
    else:
        res = sum_FaFb / (np.sqrt(sum_Fa2) * np.sqrt(sum_Fb2))
    
    return res 
