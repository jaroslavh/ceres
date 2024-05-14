import numpy as np
import pynndescent
import logging


class ReverseNeighborhood:
    def __init__(self):
        # reverse neighborhoods of all nodes in dataset - {node1: {neighbor2, neighbor2}, node2: ...}
        self.reverse_neighborhood = dict()

    def add_neighbor(self, node: int, neighbor: int):
        if node not in self.reverse_neighborhood.keys():
            self.reverse_neighborhood[node] = set()
        self.reverse_neighborhood[node].add(neighbor)

    # TODO make it numpy


def highest_count_index_generator(arr):
    for val in np.flip(np.argsort(arr)):
        yield val


def select_representatives_for_one_class(datapoints, distance_threshold, coverage, sample_rate,
                                         dist_func=None, pynndescent_k=10):
    # build index
    pruning_degree_multiplier = len(datapoints) / pynndescent_k
    index = pynndescent.NNDescent(datapoints,
                                  n_neighbors=pynndescent_k,
                                  metric=dist_func,
                                  pruning_degree_multiplier=pruning_degree_multiplier,
                                  diversify_prob=0.0)
    index.prepare()
    # get neighbors
    neighbors = index.query(datapoints, k=pynndescent_k, epsilon=sample_rate)
    class_size = neighbors[0].shape[0]
    distances = neighbors[1][:, 1:]
    indices = neighbors[0][:, 1:]  # first index is self similarity (always 1)

    # create reverse neighborhood
    reverse_neighborhoods = ReverseNeighborhood()
    neighborhood_sizes = np.zeros(shape=len(neighbors[0]), dtype='int')
    for node_index, node_row in enumerate(indices):  # iterate over all nodes
        for row_index, neighbor_index in enumerate(node_row):  # iterate over all neighbors of indices
            if distances[node_index, row_index] < distance_threshold:  # filter only the ones in threshold
                reverse_neighborhoods.add_neighbor(neighbor_index, node_index)
                neighborhood_sizes[neighbor_index] += 1

    selected_representatives = set()
    covered = set()

    highest_coverage_generator = highest_count_index_generator(neighborhood_sizes)

    empty_neighbors_flag = False
    while len(covered) / class_size < coverage:
        pivot_index = next(highest_coverage_generator)
        if pivot_index not in covered:
            # print(f"Covered {len(covered) / class_size:4.3f} of current class.")
            selected_representatives.add(pivot_index)
            covered.add(pivot_index)
            try:
                for neighbor in reverse_neighborhoods.reverse_neighborhood[pivot_index]:
                    covered.add(neighbor)
            except KeyError:
                assert (neighborhood_sizes[pivot_index] == 0)
                if not empty_neighbors_flag:
                    empty_neighbors_flag = True
                    logging.info(
                        f"\tReached only coverage of {len(covered) / class_size:4.3f}, with {len(selected_representatives)} representatives. The rest of datapoints are considered outliers.")
                    break

    return selected_representatives
