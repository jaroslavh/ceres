import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

import freq_similarity


class Dataset(object):
    """Store training and testing parts of data in this object.
    
    :param X_train: training set sample keys for data dictionary
    :type X_train: np.ndarray
    :param X_test: testing set samples
    :type X_test: np.ndarray
    :param y_train: training set labels
    :type y_train: np.ndarray
    :param y_test: testing set labels
    :type y_test: np.ndarray
    :param classes: labels of classes present in this dataset
    :type classes: list
    """

    # NOTE REVISITED
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """Constructor method."""

        self.classes = list(set(y_test).union(set(y_train)))
        self.train = self.initialize_train_classes(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

    # NOTE REVISITED
    @classmethod
    def from_one_file(cls, samples: np.ndarray, labels: np.ndarray, test_size: float = .2) -> 'Dataset':
        # TODO each class should be split 80/20 instead of the whole dataset!
        """Create a Dataset by splitting one file.
        
        :param samples: samples from one file
        :type samples: np.ndarray
        :param labels: labels corresponding to samples
        :type labels: np.ndarray
        :param test_size: part of dataset to be split as test_size
        :type test_size: float
        :return: New dataset created from one file.
        :rtype: Dataset
        """
        # TODO random state always the same for reproducibility, change later
        return cls(*train_test_split(samples, labels, test_size=test_size, random_state=None))

    # NOTE REVISITED
    def initialize_train_classes(self, X_train, y_train):
        """Return classes with their members as a dict. e.g. classes = {'cat':[
        
        :return: classes with their identifiers {class_id:{set of members}}
        :rtype: dict
        """
        classes = dict()
        for c in self.classes:
            try:
                classes[c] = X_train[[i for i, _ in enumerate(X_train) if y_train[i] == c], :]
            except TypeError:
                classes[c] = [val for val, label in zip(X_train, y_train) if label == c]

        return classes

    # NOTE REVISITED
    def get_test_data(self):
        """Return a tuple of test samples and their labels.

        :return: (samples, labels) for test data
        :rtype: tupple
        """
        return self.X_test, self.y_test

    def get_class_train_size(self, class_id):
        """Return size of train data of given class.

        :param class_id: class identifier
        :type class_id: str
        :return: size of train class
        :rtype: int
        """
        return len(self.train[class_id])

    def get_class_size(self, class_id):
        """Return size of given class.

        :param class_id: class identifier
        :type class_id: str
        :return: size of train class
        :rtype: int
        """
        return len([i for i in self.y_train + self.y_test if i == class_id])

    # NOTE REVISITED
    def get_class_homogeneity(self, class_id: str, similarity,
                              sample_rate: float = 0.05, max_sample_size: int = 1000):
        """Return homogeneity of given class.

        This method selects 5% by default of the train data and calculates the mean of sum of full
            similarity matrix as a homogeneity of a class. The maximum sample size is set to 100
            and the minimum to 10.

        :param class_id: class identifier
        :type class_id: str
        :param similarity: similarity function with return value [0, 1]
        :type similarity: function
        :param sample_rate: from which portion of class to calculate homogeneity
        :type sample_rate: float
        :param max_sample_size: highest value of samples to be selected
        :type max_sample_size: int
        :return: estimate homogeneity of the class
        :rtype: float
        """
        # get all class samples
        class_samples = self.train[class_id]

        # determine sample size
        assert (max_sample_size != 0)
        sample_size = int(len(class_samples) * sample_rate)
        if sample_size < 10:
            sample_size = 10
        elif sample_size > max_sample_size:
            sample_size = max_sample_size

        # sample data
        if sample_size < len(class_samples):
            class_samples = [class_samples[i] for i in np.random.choice(range(sample_size), sample_size)]

        return self.__calculate_homogeneity(class_samples, similarity)

    def get_class_distance_threshold(self, class_id: str, quantile: float, similarity,
                                     sample_rate: float = 0.1, max_sample_size: int = 1000):

        # TODO limit by max_sample_size
        # get all class samples
        class_samples = self.train[class_id]
        class_size = len(class_samples)
        random_samples = [class_samples[i] for i in
                          np.random.choice(range(class_size), int(sample_rate * class_size))]

        # calculate frequencies of similarity matrix
        frequencies = []
        try:
            frequencies = np.reshape(pairwise_distances(random_samples, metric=similarity), -1)
        except:
            for i, i_val in enumerate(random_samples):
                for j, j_val in enumerate(random_samples):
                    if j <= i:
                        frequencies.append(similarity(i_val, j_val))
                    else:
                        break

        if quantile is None:
            hist = np.histogram(frequencies, bins=[x / 100.0 for x in range(0, 105, 5)])
            quantile = hist[1][np.argmax(hist[0]) + 1]
            # print(f"Histogram: {hist}")
            # print(f"Quantile for similarity threshold is set to: {quantile}")
        return np.quantile(frequencies, quantile)

    # NOTE REVISITED
    @staticmethod
    def __calculate_homogeneity(samples: np.ndarray, similarity):
        """Return estimate homogeneity of given samples.

        :param samples: train samples from given class
        :type samples: np.ndarray
        :param similarity: similarity function with return value [0, 1]
        :type similarity: function
        :return: homogeneity of the data
        :rtype: float
        """
        # calculate frequencies of similarity matrix
        frequencies = []
        for i, i_val in enumerate(samples):
            for j, j_val in enumerate(samples):
                if j <= i:
                    frequencies.append(similarity(i_val, j_val))
                else:
                    break

        samples_num = len(samples)
        return sum(frequencies) / (samples_num * (samples_num - 1))
        # return np.mean(frequencies)

    # @staticmethod
    # def calculate_full_similarity_matrix(data: np.array, sim):
    #     """Calculate full similarity matrix from given data.
    #
    #     :param data: 2d array of samples that can be given to sim function and return a value
    #     :type data: np.ndarray
    #     :param sim: similarity function that takes as argument 2 samples from data
    #     :type sim: function
    #     """
    #     if sim == pynndescent.distances.euclidean:
    #         print("NOTE: Using slow similarity matrix computation for euclidean distance.")
    #         matrix = np.ndarray(shape=(data.shape[0], data.shape[0]), dtype=float)
    #         for i, sample_i in enumerate(data):
    #             matrix[i][i] = 0.0
    #             for j, sample_j in enumerate(data[:i]):
    #                 tmp_sim = sim(sample_i, sample_j)
    #                 matrix[i][j] = tmp_sim
    #                 matrix[j][i] = tmp_sim
    #         print("Similarity Matrix created.")
    #     else:
    #         matrix = squareform(pdist(data, sim))
    #     return matrix

    def get_class_full_matrix(self, class_id, sim):
        if sim == freq_similarity.freq_sim:
            data = self.train[class_id]
            matrix = np.ndarray(shape=(len(data), len(data)), dtype=float)
            for i, sample_i in enumerate(data):
                matrix[i][i] = 0.0
                for j, sample_j in enumerate(data[:i]):
                    tmp_sim = sim(sample_i, sample_j)
                    matrix[i][j] = tmp_sim
                    matrix[j][i] = tmp_sim
            return matrix
        else:
            # matrix = self.calculate_full_similarity_matrix(class_samples, sim)
            return pairwise_distances(self.train[class_id], metric=sim)
