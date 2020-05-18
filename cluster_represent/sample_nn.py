from .heap_item import heapItem
import heapq

class SampleNN(object):
    """Represents one sample in arbitarary space

    :param K: Size of the heap
    :type K: int
    :param name: Unique identifier of the object
    :type name: int
    :param values: Objects coordinates in arbitrary space. 
    :type values: list
    :param similarity: Function that takes 2 lists and returns [0,1] value.
    :type similarity: function
    :param heap: heapItem objects stored in a heap
    :type heap: list of heapItems
    :param unique: set of names of heapItems stored in the heap
    :type unique: set
    :param in_samples: Samples to initate heap with.
    :type in_samples: list of K samples
    :raises ValueError: Raised when in_samples is longer than K.
    """

    def __init__(self, K, name, values, in_samples, similarity):
        """Constructor method."""
        
        if K < len(in_samples):
            raise ValueError('Input array len {0} is bigger than K ({1})'.format(in_samples.size, K))
        self.K = K
        self.name = name
        self.values = values
        self.similarity = similarity
        self.heap = []
        self.unique = set()
        for index, row in enumerate(in_samples):
            if index == self.name:
                continue
            self.heap.append(heapItem(index, 0, flag=True))
            self.unique.add(index)

    def updateNN(self, name, dist):
        """Update the heap with new item.
        
        :param name: identifier of new item
        :type name: int
        :param dist: similarity between new item and this object
        :type dist: float
        :return: Return 1 if sample was added to the heap, 0 otherwise.
        :rtype: int
        """
        if self.name == name or name in self.unique:
            return 0
        if len(self.heap) < self.K: #TODO optimiztion possible> this check gets executed every time
            self.unique.add(name)
            heapq.heappush(self.heap, heapItem(name, dist))
        else:
            if dist <= self.heap[0].similarity:
                return 0
            self.unique.remove(self.heap[0].name)
            heapq.heapreplace(self.heap, heapItem(name, dist))
            self.unique.add(name)
        return 1

    def heapSum(self):
        """Return the sum of similarities of heap.
        
        :return: Sum of similarities in the heap.
        :rtype: float
        """
        retSum = 0
        for i in self.heap:
            retSum += i.similarity
        return retSum

    def heapMax(self):
        """Return maximal similarity present in heap.
        
        :return: maximum of similarities in the heap
        :rtype: float"""
        return max([i.similarity for i in self.heap])

    def getSim(self, name):
        """Returns simililarity with given sample.
        
        :param name: identifier of the sample
        :type name: int
        :raises KeyError: Raised if the sample is not present in the heap.
        :return: similarity between this and sample object
        :rtype: float
        """
        ret = [i.similarity for i in self.heap if i.name == name]
        if not ret:
            raise KeyError(name)
        return ret[0]
