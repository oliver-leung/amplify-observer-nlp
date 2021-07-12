import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel


class VectorSimilarity(BaseEstimator):
    _Vectors = None
    _labels = None

    def __init__(self, n_best=10):
        self.n_best = n_best

    def fit(self, X, y):
        """

        @param X:
        @param y:
        @return:
        """
        if X.dtype == np.dtype('complex128'):
            raise ValueError("Complex data not supported")

        X, y = self._validate_data(X, y)

        self._Vectors = X
        self._labels = y

        return self

    def predict(self, X):
        gram = linear_kernel(X, self._Vectors)
        gram = gram.argsort()
        outs = self._labels.take(gram[:, :self.n_best])
        return outs
