class Result(object):
    """Use to store results of experiment from Experiment class.
        
    :param algorithm: for which alorithm is the result
    :type algorithm: function
    :param parameters: with what parameters was the algorithm run
    :type labels: list
    """

    def __init__(self, algorithm, parameters):
        """Constructor method"""
        self.algorithm = algorithm
        self.parameters = parameters
        # prototypes # TODO make it dict and retrieve for classification
        self.samples = []
        self.labels = []
        self.scan_rates = {}

    def add_cluster_prototype(self, samples: list, label: int, scan_rate: float = None):
        """Add cluster prototype to result.
        
        :param samples: list of samples to be added to the Result
        :type samples: list
        :param label: identifier of the cluster
        :type label: int
        """
        self.samples += samples
        self.labels += [label] * len(samples)
        self.scan_rates[label] = scan_rate
    
