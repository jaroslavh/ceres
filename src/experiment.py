import time

from src.dataset import Dataset
from src.result import Result
import src.algorithms as algorithms


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

    def __init__(self, dataset: Dataset, algorithms: list,
                 params: list, distance_thresholds: dict):
        """Constructor method"""

        self.dataset = dataset
        self.algorithms = algorithms
        self.algorithm_params = params
        self.results = [None] * len(self.algorithms)
        self.distance_thresholds = distance_thresholds

    def run(self, logger):
        """Run the experiment that was set up.

        :return: list of Results for each algorithm in algorithms dataset
        :type: list
        """

        results = []
        for algorithm_func, params in zip(self.algorithms, self.algorithm_params):
            results.append(Result(algorithm_func.__name__, params))
            logger.info(f"\t\tAlgorithm: {algorithm_func.__name__}")
            for class_id in self.dataset.classes:  # TODO  remove comment class_samples in self.dataset.train.items():
                t0 = time.time()

                # specific handling for DS3 algorithm
                if algorithm_func == algorithms.ds3:
                    data = self.dataset.get_class_full_matrix(class_id, params['similarity'])
                    logger.info("\t\t\tFull-similarity matrix completed. DS3 starting.")
                    prototype_indices = algorithm_func(data, len(data))
                elif algorithm_func == algorithms.random_select:
                    prototype_indices = algorithm_func(self.dataset.train[class_id],  **params)
                # all other algorithms
                else:
                    prototype_indices = algorithm_func(self.dataset.train[class_id],
                                                       max_distance=self.distance_thresholds[class_id],
                                                       **params)

                prototype = [self.dataset.train[class_id][i] for i in prototype_indices]
                results[-1].add_cluster_prototype(prototype, class_id)
                t1 = time.time()
                logger.info(f"\t\t\tPrototype size {class_id}: {len(prototype)}, Time {t1 - t0}")
        return results
