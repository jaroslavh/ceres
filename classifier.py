import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


class KnnClassifier(object):
    """Class that implements k-nearest neighbor classifier for arbitrary
           similarity measures."""
    def __init__(self, similarity, K: int):
        """Parameters:
               similarity: function that takes 2 arrays and outputs a value
                   in range 0 and 1
               K: integer value of how many neighbors I am looking at while
                   classifying"""
        self.similarity = similarity
        self.K = K
        
    def classify(self, prototypes, test_data):
        """Assumes prototypes to be a tuple of (prototypes, labels)
               and prototypes to be a typle of (test samples, labels).
           This is slow brute force implementation iterating over all.
           Returns confusion matrix."""

        X_train = prototypes[0]
        y_train = prototypes[1]
        X_test = test_data[0]
        y_test = test_data[1]

        shape = len(set(y_test))
        conf_matrix = np.zeros((shape, shape), dtype=int)

        # convert cluster labels to rows and remember the order
        labels_to_int = {}
        row_labels = []
        for i, item in enumerate(set(y_test)):
            labels_to_int[item] = i
            row_labels.append(item)

        # find the most similar sample
        for i, row in enumerate(X_test):
            best_sim = 0
            for j, ref in enumerate(X_train):
                tmp_sim = self.similarity(row, ref)
                if tmp_sim > best_sim:
                    best_sim = tmp_sim
                    best_cluster = y_train[j]
                    if best_sim == 1:
                        break
                
            if best_sim == 0:  # sample cannot be classified
                print('Sample cannot be classified.')
                continue
            else:
                conf_matrix[labels_to_int[y_test[i]]][labels_to_int[best_cluster]] += 1
        
        return conf_matrix  # , row_labels

    def classify_dissimilarity(self, prototypes, test_data):
        """Assumes prototypes to be a tuple of (prototypes, labels)
               and prototypes to be a typle of (test samples, labels).
           This is slow brute force implementation iterating over all.
           Returns confusion matrix."""

        X_train = prototypes[0]
        y_train = prototypes[1]
        X_test = test_data[0]
        y_test = test_data[1]

        shape = len(set(y_test))
        conf_matrix = np.zeros((shape, shape), dtype=int)

        # convert cluster labels to rows and remember the order
        labels_to_int = {}
        row_labels = []
        for i, item in enumerate(set(y_test)):
            labels_to_int[item] = i
            row_labels.append(item)

        # find the most similar sample
        for i, row in enumerate(X_test):
            best_sim = 2.0
            for j, ref in enumerate(X_train):
                tmp_sim = self.similarity(row, ref)
                if tmp_sim < best_sim:
                    best_sim = tmp_sim
                    best_cluster = y_train[j]
                    if best_sim == 0:
                        break

            if best_sim == 2.0:  # sample cannot be classified
                print('Sample cannot be classified.')
                continue
            else:
                conf_matrix[labels_to_int[y_test[i]]][labels_to_int[best_cluster]] += 1

        return conf_matrix  # , row_labels

    def classifySklearn(self, prototypes, test_data):
        """Assumes prototypes to be a tuple of (prototypes, labels) and \
               and prototypes to be a typle of (test samples, labels).
           This implementation uses scikit-learn implementation of kNN
               classifier.
           Returns confusion matrix."""
        
        X_train = prototypes[0]
        y_train = prototypes[1]
        X_test = test_data[0]
        y_test = test_data[1]

        classifier = KNeighborsClassifier(n_neighbors=self.K, metric=self.similarity, n_jobs=-1)
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)

        return conf_mat
