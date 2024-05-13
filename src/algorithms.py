from collections import defaultdict

import numpy as np
import src.crs as crs

import matlab.engine
import logging
import src.nndescent as nndescent


def random_select(samples: np.ndarray, select: float = 0.05):
    """Assumes df_index to be a set and num to be an int.
       Returns random subset of size n from input set of indices."""
    number_to_select = int(np.ceil(len(samples) * select))
    indices = np.random.choice(list(range(len(samples))), number_to_select)
    return indices


def delta_medoids(samples: np.ndarray, distance_function, max_distance: float):
    t = 0  # iteration number
    representatives = {0: set()}

    while True:
        logging.info(f'\t\t\titeration:{t}, reps:{len(representatives[t])}')
        t += 1
        clusters = defaultdict(set)
        # repAssign routine
        for index, val in enumerate(samples):
            dist = np.inf
            representative = None
            for rep in clusters.keys():  # finding correct neighborhood
                tmp_dist = distance_function(val, samples[rep])
                if tmp_dist < dist:
                    representative = rep
                    dist = tmp_dist
            if dist < max_distance:  # located existing neighborhood
                clusters[representative].add(index)
            else:  # creating new neighborhood
                clusters[index] = {index}

        representatives[t] = set()

        for cluster in clusters.values():
            # finding best representative
            min_sum = np.inf
            best_rep = None
            for i in cluster:  # argmin on distances in cluster
                dist_sum = 0
                for j in cluster:
                    tmp_dist = distance_function(samples[i], samples[j])
                    if tmp_dist < max_distance:
                        dist_sum += tmp_dist
                if dist_sum < min_sum:
                    min_sum = dist_sum
                    best_rep = i

            representatives[t].add(best_rep)

        if representatives[t] == representatives[t - 1]:
            break
        # if len(representatives) > 2 and representatives[t] == representatives[t - 2]:
        #     break
    return representatives[t]


def nndescent_reverse_neighbors(samples, coverage: float,
                                sample_rate: float,
                                dist_func,
                                K: int = None,
                                max_distance: float = 0.6):
    representatives = crs.select_representatives_for_one_class(samples,
                                                               distance_threshold=max_distance,
                                                               coverage=coverage,
                                                               sample_rate=sample_rate,
                                                               dist_func=dist_func,
                                                               pynndescent_k=K)
    return representatives


def custom_nndescent_reverse_neighbors(samples, coverage: float,
                                       sample_rate: float,
                                       similarity,
                                       K: int = None,
                                       min_similarity: float = 0.6):
    knn_graph = nndescent.NNDescentFull(dataset=samples,
                                        similarity=similarity,
                                        K=K,
                                        sample_rate=sample_rate)

    print(min_similarity)
    representatives = nndescent.getReprIndicesReverseNeighborsThreshold(knn_graph, coverage, min_similarity)
    return representatives


# def ds3(matrix: np.ndarray, shape: int):
#     """Wrapper for calling ds3 in matlab.
#
#     :param matrix: square matrix of shape*shape
#     :type matrix: np.ndarray
#     :param shape: int
#     :type shape: size of matrix
#     :return: int
#     :rtype: matlab.double or float (float if only one representative is selected
#     """
#     eng = matlab.engine.start_matlab()
#     eng.cd('DS3_v1.0')
#     matrix = matrix.flatten().tolist()
#
#     M = eng.cell2mat(matrix)
#     ret = eng.func_run_ds3(M, shape)
#     eng.quit()
#     # transform results from float to int and subtract 1 -> matlab to python index correction
#     if type(ret) == float:
#         logging.info(f"Found only one representative {ret}")
#         return [int(ret) - 1]
#     return [int(i) - 1 for i in ret[0]]
