from sklearn.base import BaseEstimator, ClusterMixin

class CosinePredictor(BaseEstimator, ClusterMixin):
    def __init__(self, test='test'):
        self.test = test
        
    def get_params(self, deep=True):
        return {'test': self.test}
    
    def fit(self, X, y=None):
        pass