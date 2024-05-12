import time

from dataset import Dataset
from result import Result
import algorithms


class Experiment(object):
    """Wrapper for running algorithms on dataset with given parameters.

    :param dataset: the dataset the experiment is conducted on
    :type dataset: Dataset
    :param coverage: desired coverage of the dataset by each method
    :type coverage: float
    :param algorithms: algorithms in order to be conducted
    :type algorithms: list
    :param params: params for algorithms in the algorithms parameter, order needs to be the same
    :type params: list
    :param results: Result for each experiment in an order they were conducted
    :type results: list
    """

    def __init__(self, dataset: Dataset, coverage: float, algorithms: list,
                 params: list, homogeneities: dict):
        """Constructor method"""

        self.dataset = dataset
        self.coverage = coverage
        self.algorithms = algorithms
        self.algorithm_params = params
        self.results = [None] * len(self.algorithms)
        self.homogeneities = homogeneities

    def run(self):
        """Run the experiment that was set up.

        :return: list of Results for each algorithm in algorithms dataset
        :type: list
        """

        results = []
        for algorithm_func, params in zip(self.algorithms, self.algorithm_params):
            results.append(Result(algorithm_func, params))
            print('Algorithm:', algorithm_func)
            for class_id in self.dataset.classes:  # TODO  remove comment class_samples in self.dataset.train.items():
                print('  Class:      ', class_id)
                t0 = time.time()

                # specific handling for DS3 algorithm
                if algorithm_func == algorithms.ds3:
                    data = self.dataset.get_class_full_matrix(class_id, params['similarity'])
                    print("    Full-similarity matrix completed. DS3 starting.")
                    prototype_indices = algorithm_func(data, len(data))
                # all other algorithms
                else:
                    prototype_indices = algorithm_func(self.dataset.train[class_id], self.coverage, **params,
                                                       threshold=self.homogeneities[class_id])

                prototype = [self.dataset.train[class_id][i] for i in prototype_indices]
                results[-1].add_cluster_prototype(prototype, class_id)
                t1 = time.time()
                print(f"Prototype size: {len(prototype)},    Time       {t1 - t0}")
        return results
