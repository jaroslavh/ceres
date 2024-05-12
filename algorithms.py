import numpy as np
import crs

import matlab.engine

import nndescent


def random_select(samples: np.ndarray, coverage: float, select: float = 0.05, threshold=None):
    """Assumes df_index to be a set and num to be an int.
       Returns random subset of size n from input set of indices."""
    number_to_select = int(np.ceil(len(samples) * select))
    indices = np.random.choice(list(range(len(samples))), number_to_select)
    return indices


def delta_one_shot(samples: np.ndarray, coverage: float, delta: float, similarity):
    """Goes once through data starting at first point, it selects as
           representatives the points that are not close to a representative
           previously selected. Thus, it is not optimal, however it always goes
           through the data only once.
       Returns np.ndarray of selected representatives."""
    representatives = set()

    for index, val in enumerate(samples):
        if index > len(samples) * coverage:
            break

        for rep in representatives:
            if similarity(val, samples[rep]) >= delta:
                break
        else:
            representatives.add(index)

    # return [samples[i] for i in representatives]
    return representatives


def delta_medoids(samples: np.ndarray, coverage: float, dist, threshold: float):
    t = 0  # iteration number
    representatives = {0: set()}

    while True:
        print('        t =', t, len(representatives[t]), ' number of representatives at iteration t')
        t += 1
        neighborhoods = dict()  # neighborhoods inside a cluster - consists of indices
        for rep in representatives[t - 1]:
            neighborhoods[rep] = {rep}

        for index, val in enumerate(samples):
            sim = np.inf
            representative = None
            for rep in neighborhoods.keys():  # finding correct neighborhood
                tmp_sim = dist(val, samples[rep])
                if tmp_sim < sim:
                    representative = rep
                    sim = tmp_sim
            if sim < threshold:  # located existing neighborhood
                neighborhoods[representative].add(index)
            else:  # creating new neighborhood
                neighborhoods[index] = {index}

        representatives[t] = set()
        for val in neighborhoods.values():
            # finding best representative
            min_sum = np.inf
            best_rep = None
            for i in val:
                dist_sum = 0
                for j in val:
                    dist_sum += dist(samples[i], samples[j])
                if dist_sum < min_sum:
                    min_sum = dist_sum
                    best_rep = i

            representatives[t].add(best_rep)
        if representatives[t] == representatives[t - 1]:
            break
        if len(representatives) > 2 and representatives[t] == representatives[t - 2]:
            break
    return representatives[t]


def nndescent_reverse_neighbors(samples, coverage: float,
                                sample_rate: float, similarity, K: int = None, threshold: float = 0.6):
    if K == None:
        sample_size = len(samples)
        # K = int(np.ceil(np.log(len(samples) + 1))) + 2                        #A
        # K = int(np.ceil(np.sqrt(len(samples) + 1) * (1 - threshold)))         #B
        # K = int(np.sqrt(len(samples)/np.log(len(samples)))                    #C
        # K = int(np.sqrt(len(samples)/np.log(len(samples))) * (1 - threshold))  #D
        K = int(np.sqrt(sample_size / (np.log2(sample_size) * 2)) * (1 + (1 - threshold)))  # E
        if K < 5:
            K = 5

    print('    K set to:', K)
    representatives = crs.select_representatives_for_one_class(samples,
                                                               similarity_threshold=threshold,
                                                               coverage=coverage,
                                                               sample_rate=sample_rate,
                                                               similarity_func=similarity,
                                                               pynndescent_k=K)
    return representatives


def custom_nndescent_reverse_neighbors(samples, coverage: float,
                                       sample_rate: float, similarity, K: int = None, threshold: float = 0.6):
    if K == None:
        sample_size = len(samples)
        # K = int(np.ceil(np.log(len(samples) + 1))) + 2                        #A
        # K = int(np.ceil(np.sqrt(len(samples) + 1) * (1 - threshold)))         #B
        # K = int(np.sqrt(len(samples)/np.log(len(samples)))                    #C
        # K = int(np.sqrt(len(samples)/np.log(len(samples))) * (1 - threshold))  #D
        K = int(np.sqrt(sample_size / (np.log2(sample_size) * 2)) * (1 + (1 - threshold)))  # E
        if K < 5:
            K = 5

    print('    K set to:', K)
    knn_graph = nndescent.NNDescentFull(dataset=samples,
                                        similarity=similarity,
                                        K=K,
                                        sample_rate=sample_rate)

    representatives = nndescent.getReprIndicesReverseNeighborsThreshold(knn_graph, coverage, threshold)
    return representatives

def ds3(matrix: np.ndarray, shape: int):
    """Wrapper for calling ds3 in matlab.

    :param matrix: square matrix of shape*shape
    :type matrix: np.ndarray
    :param shape: int
    :type shape: size of matrix
    :return: int
    :rtype: matlab.double or float (float if only one representative is selected
    """
    eng = matlab.engine.start_matlab()
    eng.cd('DS3_v1.0')
    matrix = matrix.flatten().tolist()

    M = eng.cell2mat(matrix)
    ret = eng.func_run_ds3(M, shape)
    eng.quit()
    # transform results from float to int and subtract 1 -> matlab to python index correction
    if type(ret) == float:
        print(f"Found only one representative {ret}")
        return [int(ret) - 1]
    return [int(i) - 1 for i in ret[0]]