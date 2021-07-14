import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel


class VectorSimilarity(BaseEstimator):
    _Vectors = None
    _labels = None

    def __init__(self, n_best=10):
        self.n_best = n_best

    def fit(self, X, y):
        # Required to pass check_estimator()
        if X.dtype == np.dtype('complex128'):
            raise ValueError('Complex data not supported')
            
        # Convert dense (i.e. "efficient") array representation to sparse
        if not isinstance(X, (np.ndarray, np.generic)):
            X = X.toarray()
            
        X, y = self._validate_data(X, y)

        self._Vectors = X
        self._labels = y

        return self

    def predict(self, X):
        gram_matrix = linear_kernel(X, self._Vectors)
        gram_descending = np.flip(gram_matrix.argsort(), axis=1)

        n_best_labels = self._labels.take(gram_descending[:, :self.n_best])
        n_best_confidence = gram_descending.take(gram_descending[:, :self.n_best])

#         return n_best_labels, n_best_confidence

        print(n_best_labels.shape)
        print(X.shape)
        return n_best_labels
