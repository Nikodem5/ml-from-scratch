import numpy as np
from sklearn.preprocessing import StandardScaler

class ZScoreScaler:
    def __init__(self, epsilon=1e-8):
        self.mean_ = None
        self.scale_ = None
        self.epsilon = epsilon

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + self.epsilon
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    