import numpy as np
import pandas as pd
import csv
import json

from src.dataset import Dataset
import sklearn.model_selection
from sklearn import datasets
from src.similarities import create_numba_dict





# NOTE REVISITED
def read_csv(filename: str):
    """Reads a file and passes it to Dataset object.
       Params:
           filename: string, path to a csv file
       Returns:
           Dataset, created from the loaded file"""
    data = np.genfromtxt(filename, delimiter=',')
    samples = data[:, :-1]
    labels = data[:, -1].astype(int)
    return Dataset.from_one_file(samples, labels)


def read_npy(filename: str):
    with open(filename, 'rb') as f:
        data = np.load(f)
    samples = data[:, :-1]
    labels = data[:, -1].astype(int)
    return Dataset.from_one_file(samples, labels)


def readTrainTestFile(train_file: str, test_file: str):
    """Loads data from 2 files as many datasets provide them.
       Params:
           train_file: string, path to train data
           test_file: string, path to test data
       Returns:
           Dataset, created from loaded files"""
    train_data = np.genfromtxt(train_file, delimiter=',')
    test_data = np.genfromtxt(test_file, delimiter=',')

    X_train = list(range(len(train_data)))
    y_train = list(train_data[:, -1].astype(int))
    X_test = range(len(test_data))
    X_test = [i + len(X_train) for i in range(len(test_data))]  # after X_train
    y_test = list(test_data[:, -1].astype(int))
    data = {}
    for i, val in enumerate(list(train_data[:, :-1]) + list(test_data[:, :-1])):
        data[i] = val

    return Dataset(X_train, X_test, y_train, y_test, data)


def readNetworkFile(batch_folder: str, time_range: int, label_file: str):
    """Create a dataset object from network batches.
    
    :param batch_folder: folder with batch files
    :type batch_folder: str
    :param time_range: number of batches to be loaded
    :type time_range: int
    :param label_file: file with labels of cluster hosts
    :type label_file: str
    :return: dataset object with data loaded from given range
    :rtype: dataset.Dataset
    """

    loaded_data = loadBatchFolderTimeRange(batch_folder=batch_folder, time_range=time_range)
    loaded_labels = readLabelFile(path=label_file)

    labels = list()
    samples = list()

    for record_label in loaded_data:
        if record_label in loaded_labels.keys():
            labels.append(loaded_labels[record_label])
            samples.append(record_label)

    return Dataset.from_one_file(samples, labels)


def createDatasetFromDataAndLabels(data: dict, in_labels: dict):
    """Create dataset from data and labels.
    
    :param data: data in the following structure {user1:{host1:freq, host2:freq}, user2:{host1:freq, host2:freq}}
    :type data: dict
    :param labels: labels of given data {key0: labelA, key1: labelB}
    :type labels: dict
    :return: Dataset object created from input data
    :rtype: Dataset
    """

    labels = list()
    samples = list()

    for key in data:
        if key in in_labels.keys():
            labels.append(in_labels[key])  # label of given sample
            samples.append(key)  # key of given sample

    return Dataset.from_one_file(samples, labels, data)


def readLabelFile(path: str):
    """Read csv file with labels of hosts. Each line is expected have the following structure:
       host(str):community_label(int)
    
    :param path: Path of csv file.
    :type path: str
    :return: labels assigned to given hosts
    :rtype: dict
    """
    with open(path, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')
        labels = {row[1]: row[0] for row in reader}

    return labels


def loadBatchFolderTimeRange(batch_folder: str, time_range: int):
    """Load time range of caught data in from given folder.
    
    :param batch_folder: path to folder
    :type batch_folder: str
    :param time_range: number of time windowes
    :type time_range: int
    :return: data loaded into this structure {user1:{host1:freq, host2:freq}, user2:{host1:freq, host2:freq}}
    :rtype: dictionary
    """

    loaded_data = {}
    for i in range(time_range):
        batch_file_path = ''.join([batch_folder, 'batch_', str(i), '.tsv'])

        data_file = pd.read_csv(batch_file_path, sep='\t')
        # structure to load information from file into
        # {user1:{host1:freq, host2:freq}, user2:{host1:freq, host2:freq}}
        for _, row in data_file.iterrows():
            # extract information from hostnamePort
            raw_hostnames = row['hostnamePort'].split(';')
            user_id = row['userID']

            # update tables of userIDs
            if user_id not in loaded_data:
                hostnames = {}
                for host in raw_hostnames:
                    hostnames[host] = 1
                loaded_data[user_id] = hostnames
            else:
                for host in raw_hostnames:
                    if host in loaded_data[user_id]:
                        loaded_data[user_id][host] += 1
                    else:
                        loaded_data[user_id][host] = 1

    return loaded_data


def loadJSON(path: str):
    """Load dumped data file from JSON created ny data_loader.
    
    :param path: path of json file
    :type path: str
    :return: data in the following structure {user1:{host1:freq, host2:freq}, user2:{host1:freq, host2:freq}}
    :rtype: dictionary
    """
    # assumes data was saved by data_loader.saveJSON method
    with open(path, 'r') as f:
        loaded_data = json.load(f)
    return loaded_data


def saveJSON(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_network_data_from_json(path: str):
    """Load dumped data file from JSON created by data_loader.

    :param path: path of json file
    :type path: str
    :return: data in the following structure {user1:{host1:freq, host2:freq}, user2:{host1:freq, host2:freq}}
    :rtype: dictionary
    """
    with open(path, 'r') as f:
        loaded_data = json.load(f)
    return loaded_data


def read_network_labels_from_csv(path: str):
    """Read csv file with labels of hosts. Each line is expected have the following structure:
       host(str):community_label(int)

    :param path: Path of csv file.
    :type path: str
    :return: labels assigned to given hosts
    :rtype: dict
    """
    with open(path, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')
        labels = {row[1]: row[0] for row in reader}
    return labels


def get_real_network_data(sample_path, label_path):
    sample_dict = load_network_data_from_json(sample_path)
    labels = read_network_labels_from_csv(label_path)

    X = []
    y = []
    no_label = 0
    for host_id, visits_dict in sample_dict.items():
        if host_id in labels.keys():
            X.append(visits_dict)
            y.append(labels[host_id])
        else:  # items that belong to cluster smaller than 30
            no_label += 1

    return X, y


def get_internet_advertisements(path: str):
    dataset = pd.read_csv(path)
    X = dataset['jacc'].tolist()
    y = []
    for label in dataset['label']:
        if label == 'nonad.':
            y.append(0)
        else:
            y.append(1)
    return X, y


def get_dataset(name: str, path: str, random_state: int = 42, test_size=0.1) -> Dataset:
    if name == '20newsgroups':
        X, y = datasets.fetch_20newsgroups_vectorized(subset="all", return_X_y=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
        return Dataset(X_train.toarray(), X_test.toarray(), y_train, y_test)
    elif name == 'network':
        X, y = get_real_network_data(f'{path}/data_288_windows.json', f'{path}/labels.tsv')
        X = [[(k, v) for k, v in item.items()] for item in X]
        X = [create_numba_dict(i) for i in X]
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
        return Dataset(X_train, X_test, y_train, y_test)
    elif name == 'mnist-fashion':
        import src.mnist_reader as mnist_reader
        # loading training data
        X, y = mnist_reader.load_mnist(path, kind='train')
        # loading testing data - reducing size for speed :)
        X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X, y,  test_size=test_size, random_state=random_state)
        X_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
        return Dataset(X_train, X_test, y_train, y_test)
    elif name == 'internet-advertisements':
        cols = pd.read_csv(f"{path}/ad.names", header=None)[0]
        x = pd.read_csv(f"{path}/ad.data", sep=",", dtype=str, header=None,names=cols)
        df = x.iloc[:, 3:]
        df = df.replace('?', None).dropna()  # removing invalid values
        X = df.loc[:, df.columns != 'label'].to_numpy().astype(int)
        y = df['label'].tolist()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
        return Dataset(X_train, X_test, y_train, y_test)
    else:
        print(f"Unknown dataset name '{name}'.")