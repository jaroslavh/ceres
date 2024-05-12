# Implementation of NNDescent algorithm for fast kNN graph construction with generic
# similarity measure.
# src: http://www.ambuehler.ethz.ch/CDstore/www2011/proceedings/p577.pdf

import random
import numpy as np
import logging

from src.sample_nn import SampleNN

def sample(dataset: list, n: int):
    """Return random sample from the whole dataset.

    :param dataset: whole dataset
    :type: list
    :param n: size of sample
    :type: int
    :return: list of random samples from the whole dataset
    :rtype: list"""
    indices = np.random.choice(range(len(dataset)), n, replace=False)
    return [dataset[i] for i in indices]

def sampleNames(name_set: set, n: int):
    """Randomly sample n names from name set.
    
    :param name_set: set of names
    :type name_set: set
    :param n: number of names to be sampled
    :type n: int
    :return: randomly sampled set of names of size n
    :rtype: set
    """
    if n > len(name_set):
        return name_set
    return random.sample(name_set, n)

def reverseSetOld(B):
    """Assumes B a collection of sets.
       Returns reverse of those sets."""
    B_keys = B.keys()
    #this is nice but not optimal, ~40% of time is spent here
    return {v:{i for i in B_keys if v in B[i]} for v in B_keys} 

def reverseSet(B):
    ret = {v:set() for v in B.keys()}

    for v in B.keys():
        for n in B[v]:
            ret[n].add(v)
    return ret

def NNDescentFull(dataset, similarity, K, sample_rate):
    """Assumes dataset a pandas.Dataframe, similarity a similarity measure function with 2 arguments,
          K a positive integer, sample_rate in (0, 1], delta a float.
       Returns a kNN graph of the given dataset."""

    # initialize kNN values
    B = {}
    for index, row in enumerate(dataset):
        B[index] = SampleNN(K=K, name=index, values=row, in_samples=sample(dataset, min(K, len(dataset))),
                            similarity=similarity)
    
    # scan = 0  # how many times distance metric is computed
    scan = np.zeros((len(dataset), len(dataset)))
    scan[:] = np.nan
    all_scan = 0
    old, new = {}, {}
    while True:
        for v in B.keys():
            old[v] = {i.name for i in B[v].heap if i.flag is False}
            sizeCounter = 0
            new[v] = set()  # TODO select nodes labeled True more efficiently
            for i, j in enumerate(B[v].heap):
                if j.flag is True:
                    new[v].add(j.name)
                    j.flag = False
                    sizeCounter += 1
                if sizeCounter >= sample_rate*K:
                    break
        oldR = reverseSet(old)
        newR = reverseSet(new)
        c = 0
        for v in B.keys():
            old[v] = old[v].union(sampleNames(oldR[v], int(sample_rate*K)))
            new[v] = new[v].union(sampleNames(newR[v], int(sample_rate*K)))
            for u1 in new[v]:
                u2_set = old[v].union({i for i in new[v] if u1 < i}) #selection of u2
                for u2 in u2_set:
                    if (u1 in B[u2].unique) and (u2 in B[u1].unique):
                        continue
                    if u1 == u2:
                        tmpSim = 1.0
                    else:
                        if np.isnan(scan[u1, u2]):
                            tmpSim = similarity(dataset[u1], dataset[u2])
                            scan[u1, u2] = tmpSim
                            scan[u2, u1] = tmpSim
                            all_scan += 1
                        else:
                            tmpSim = scan[u1, u2]
                    c += B[u1].updateNN(name=B[u2].name, dist=tmpSim)
                    c += B[u2].updateNN(name=B[u1].name, dist=tmpSim)
        #print('C after whole run is', c)
        delta = 0.001
        if c < delta*K*len(dataset):
            upper_tri = np.triu(scan, k=1).flatten()
            number_sims = np.count_nonzero(upper_tri[~np.isnan(upper_tri)])

            logging.info(f'\t\t\tkNN scan rate {(number_sims / (len(dataset) * (len(dataset) - 1)) / 2)}, {number_sims}, {all_scan}')
            #import csv
            #with open('scan_rates_final.csv', 'a') as f: #logging progress
                #wr = csv.writer(f)
                #wr.writerow([K, (scan / (len(dataset) * (len(dataset) - 1)) / 2)])
            return B


def getReprIndicesReverseNeighborsThreshold(knn_res: dict, coverage: float, sim_threshold: float):
    """Find represetatives from a given kNN graph.

    :param knn_res: kNN graph as a result from nnDescentFull algorithm.
    :type knn_res: dict
    :param coverage: percentage of graph to be covered
    :type coverage: float
    :param sim_threshold: similarity under which the edges are not considered in reverse kNN
    :type sim_threshold: float
    :return: list of indices labels that were selected as representatives
    :rtype: list
    """
    full_len = len(knn_res)
    reverse = {}
    for key in knn_res.keys():
        reverse[key] = set()
    counts = np.zeros(shape=full_len, dtype='float')

    for key in knn_res.keys():
        for n in knn_res[key].unique:
            if knn_res[key].getSim(n) < sim_threshold:
                continue
            reverse[n].add(key)
            counts[n] += knn_res[key].getSim(n)

    selected = []
    neighbors = set()
    while len(neighbors) / full_len < coverage:
        ix = np.where(counts == max(counts))[0][0]
        if counts[ix] == -1:
            print(counts)
            break
        elif ix not in neighbors:
            selected.append(ix)
            neighbors.add(ix)
            for n in reverse[ix]:
                if n not in neighbors:
                    for f in knn_res[n].unique:
                        if (counts[f] != -1 and knn_res[n].getSim(f) >= sim_threshold):
                            counts[f] -= knn_res[n].getSim(f)
                    neighbors.add(n)
                    counts[n] = -1
        counts[ix] = -1
    return selected