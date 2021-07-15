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

        # Model performance should be decoupled from references to training data
        self._Vectors = np.copy(X)
        self._labels = np.copy(y)

        return self

    def _gram_matrices(self, X):
        gram_matrix = linear_kernel(X, self._Vectors)
        gram_desc_args = np.fliplr(gram_matrix.argsort())
        gram_desc = np.take_along_axis(gram_matrix, gram_desc_args, axis=1)

        return gram_matrix, gram_desc_args, gram_desc

    def predict(self, X):
        return self.predict_score(X)[0]

    def score(self, X, y=None):
        return self.predict_score(X)[1]

    def predict_score(self, X):
        # Ensures that any call to predict/score will require only one call to linear_kernel()
        gram_matrix, gram_desc_args, gram_desc = self._gram_matrices(X)

        pred = self._labels.take(gram_desc_args[:, :self.n_best])
        score = gram_desc[:, :self.n_best]

        return pred, score
