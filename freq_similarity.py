import numpy as np
from numba import njit
from numba.typed import Dict
from numba.core import types


@njit
def freq_sim(host_a, host_b, timewindow_num: int = 288):
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

    # no hosts in common
    if not ha.intersection(hb):
        return 2.0

    # calculating similarity
    sum_FaFb = 0.0
    sum_Fa2 = 0.0
    sum_Fb2 = 0.0
    for s in S:
        if s in ha:
            Fa = host_a[s] / timewindow_num
        else:
            Fa = 0.0
        if s in hb:
            Fb = host_b[s] / timewindow_num
        else:
            Fb = 0.0
        sum_FaFb += Fa * Fb
        sum_Fa2 += Fa ** 2
        sum_Fb2 += Fb ** 2
    if sum_FaFb == 0.0:
        return 2.0
    else:
        res = sum_FaFb / (np.sqrt(sum_Fa2) * np.sqrt(sum_Fb2))

    return 1 - res

def create_numba_dict(x: list):
    freq_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )

    for k, v in x:
        if v != None:
            freq_dict[k] = int(v)

    return freq_dict