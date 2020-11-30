import numpy as np
from pynndescent import NNDescent
import matplotlib.pyplot as plt


def create_knn_graph(samples, k, similarity):
    """Create kNN graph from given samples.

    :param samples: list of input samples
    :type samples: np.array
    :param k: number of neighbors to find
    :type k: int
    :param similarity: similarity to be used between samples
    :type similarity: function
    :return: list of neighbor lists and list of similarities to them in corresponding order
    :rtype: tuple
    """
    index = NNDescent(samples, metric=similarity)
    knn_graph = index.query(samples, k=k)
    # remove first items, they are the node itself
    neighbors = [item[1:] for item in knn_graph[0]]
    similarities = [item[1:] for item in knn_graph[1]]
    return neighbors, similarities


def find_biggest_neighbourhood_node(neigh_graph, covered):
    """Find samples from neighborhood graph with highest number of neighbors.

    :param neigh_graph: neighborhood graph dictionary with each key representing one node
    and value containing its neighbors
    :return: list of nodes with highest number of neighbors
    """
    max_size = 0
    biggest = []
    for key, val in neigh_graph.items():
        if key not in covered:
            cur_size = len(val)
            if cur_size > max_size:
                biggest = []
                max_size = cur_size
                biggest.append(key)
            elif cur_size == max_size:
                biggest.append(key)
    return biggest


def find_best_repr_candidate(nodes, reverse_neigh):
    """Given list of nodes and their similarities to their neighbors, find the best
    candidate for a representative.

    :param nodes: list of node identifiers
    :param reverse_neigh: reverse neighborhood graph
    :return: node with biggest sum of similarities
    """
    sums = np.array([sum(reverse_neigh[node].values()) for node in nodes])
    return nodes[np.where(sums == max(sums))[0][0]]


def compute_reverse_graph(neighbors, similarities):
    """Reverse graph and create a dictionary for each node with neighbors and similarities.

    :param neighbors:
    :param similarities:
    :return: dictionary of nodes with corresponding neighbors and distances to them
    :rtype: dict
    """
    # reverse graph and compute counts of occurences in the neighbourhoods
    reverse_neighbors = {i: dict() for i, _ in enumerate(neighbors)}

    for node, neigh_list in enumerate(neighbors):
        for index, neighbor in enumerate(neigh_list):
            reverse_neighbors[neighbor][node] = similarities[node][index]
    return reverse_neighbors


def find_representatives(samples, neighbors, similarities, coverage):
    """Find representatives for given set of samples.

    :param samples: Ordered list of identifiers of samples.
    :param neighbors:  Array containing list of neighbors for each sample.
    :param similarities: Array containing list of similarities to neighbors of each sample.
    :param coverage: Desired coverage of the set
    :return:
    """
    # reverse graph and compute counts of occurrences in the neighbourhoods
    reverse_neigh = compute_reverse_graph(neighbors, similarities)

    covered = set()
    representatives = set()
    pop_size = len(neighbors)
    # while the dataset is not covered enough select representatives
    while len(covered) / pop_size < coverage:
        # find samples that have the biggest reverse neighborhood and add it to representatives
        biggest = find_biggest_neighbourhood_node(reverse_neigh, covered)
        # find best representative from the biggest neighborhoods
        representative = find_best_repr_candidate(biggest, reverse_neigh)
        representatives.add(representative)
        # add the representative and its reverse neighbours to covered samples
        covered.add(representative)  # TODO check for redundancy where node 0 is itself
        covered = covered.union(reverse_neigh[representative])
        # modify the reverse neighborhood so that already covered points are removed from the graph
        for neighbor in reverse_neigh[representative]:
            for rev_neigh in neighbors[neighbor]:
                if rev_neigh != representative:
                    del reverse_neigh[rev_neigh][neighbor]
        reverse_neigh[representative] = {}
        # print(len(covered) / pop_size, sorted(representatives))
        # print_coverage_state(samples, representatives, covered) # use only with 2D data
    return representatives


def print_coverage_state(samples, reps, covs):
    """Print samples marking the representatives and already covered ones."""
    x, y = [i[0] for i in samples], [i[1] for i in samples]
    representatives = [samples[i] for i in reps]
    rep_x, rep_y = [i[0] for i in representatives], [i[1] for i in representatives]
    covered = [samples[i] for i in covs]
    cov_x, cov_y = [i[0] for i in covered], [i[1] for i in covered]
    plt.figure(figsize=(16,8))
    plt.scatter(x, y, s=50)
    plt.scatter(cov_x, cov_y, color='g', s=50)
    plt.scatter(rep_x, rep_y, color='r', s=10)
    plt.show()
