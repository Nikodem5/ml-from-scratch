import numpy as np
from ..utils.z_score_normalization import ZScoreScaler
from ..utils.activations import sigmoid

class LogisticRegression:
    def __init__(self, normalization=False):
        self.w = None
        self.b = None
        self.m = None
        self.n = None
        self.normalization = normalization
        self.scaler = ZScoreScaler() if normalization else None
    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)
    
    def _compute_cost(self):
        pass

    def _compute_gradient(self):
        pass

    def gradient_descent(self):
        pass

    def fit(self, X, y, alpha=0.01, epsilon=1e-3, iters=10, patience=5, info=False):
        self.m, self.n = X.shape
        self.w = np.random.rand(self.n)
        self.b = np.random.randint(1, 10)

        #TODO gradient and the rest