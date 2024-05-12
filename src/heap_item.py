class heapItem(object):

    def __init__(self, name, similarity, flag=True):
        """Assumes name an integer, similarity a float.
           Creates a heapItem object"""
        self.name = name
        self.similarity = similarity
        self.flag = flag

    def __lt__(self, other):
        return self.similarity < other.similarity
