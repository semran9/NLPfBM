import numpy as np

class GloVeVectorizer:
    def __init__(self, n):
        self.dir = './glove/glove.6B.' + str(n)+ 'd.txt'
        self.index = self.get_index(n)
    def fit(self, X, y=None):
        return self
    def vectorize(self, X):
        glove_vectors = []
        for word in X:
            vector = self.index.get(word, np.zeros(50))
            glove_vectors.append(vector)
        return np.array(glove_vectors)
    def get_coefs(self, word, *arr): 
        return word, np.asarray(arr, dtype='float32')
    def get_index(self, n):
        embeddings_index = dict(self.get_coefs(*o.rstrip().rsplit(' ')) for o in open(self.dir))
        return embeddings_index
    