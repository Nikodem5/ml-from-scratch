import numpy as np
from ..utils.z_score_normalization import ZScoreScaler

#TODO regularization, normalization, early stopping, adaptive alpha

class LinearRegression:
    def __init__(self, normalization=False) -> None:
        self.w = None
        self.b = None
        self.m = None
        self.n = None
        self.normalization = normalization
        self.scaler = ZScoreScaler() if normalization else None

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def _compute_cost(self, X, y, y_pred):
        cost = np.sum(np.square(y_pred - y))
        cost = cost / (2 * self.m)

        return cost

    def _compute_gradient(self, X, y, y_pred):
        diff_ = y_pred - y
        dj_db = np.sum(diff_)
        dj_dw = np.dot(X.T, diff_)

        dj_db = dj_db / self.m
        dj_dw = dj_dw / self.m

        return dj_dw, dj_db

    def gradient_descent(self, X, y, alpha, epsilon, iters, patience, info):
        convergance_count = 0
        for i in range(iters):
            y_pred = self.predict(X)
            dj_dw, dj_db = self._compute_gradient(X, y, y_pred)

            w_tmp = self.w - (alpha * dj_dw)
            b_tmp = self.b - (alpha * dj_db)
            self.w = w_tmp
            self.b = b_tmp

            cost = self._compute_cost(X, y, y_pred)
            if cost <= epsilon:
                convergance_count += 1
                if convergance_count >= patience:
                    if info:
                        print(f'last iteration: {i} cost: {cost}')
                    break
            else:
                convergance_count = 0

            if info and (i % (iters//10) == 0 or i + 1 == iters):
                print(f'iteration: {i} cost: {cost}')

        return self.w, self.b

    def fit(self, X, y, alpha=0.01, epsilon=1e-3, iters=10, patience=5, info=False):
        self.m, self.n = X.shape
        self.w = np.random.rand(self.n)
        self.b = np.random.randint(1, 10)

        if self.normalization:
            X = self.scaler.fit_transform(X)
            y = self.scaler.fit_transform(y)
        
        self.w, self.b = self.gradient_descent(X, y, alpha, epsilon, iters, patience, info)