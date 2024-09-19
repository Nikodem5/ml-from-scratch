import numpy as np

class LinearRegression:
    def __init__(self, ) -> None:
        self.w = None
        self.b = None
        self.m = None
        self.n = None

    def predict(self, x):
        return np.dot(self.w, x) + self.b
    
    def compute_cost(self, X, y):
        total_error = 0
        for i in range(self.m):
            total_error += np.square(self.predict(X[i] - y[i]))
        total_error = total_error / (2 * self.m)
        
        return total_error
    
    def compute_gradient(self, X, y):
        dj_db = 0
        dj_dw = np.zeros(self.n)

        for i in range(self.m):
            dj_db += self.predict(X[i]) - y[i]
            for j in range(self.n):
                dj_dw[j] += (self.predict(X[i]) - y[i]) * X[i][j]
            
        dj_db = dj_db / self.m
        dj_dw = dj_dw / self.m

        return dj_dw, dj_db
    
    def gradient_descent(self, X, y, alpha, iters):
        for i in range(iters):
            dj_dw, dj_db = self.compute_gradient(X, y)

            w_tmp = self.w - dj_dw
            b_tmp = self.b - dj_db
            w = w_tmp
            b = b_tmp

            if iters % (iters//10) == 10:
                print(f'iteration: {i} cost: {self.compute_cost(X, y)}')
        
        return self.w, self.b
    
    def fit(self, X, y, alpha=0.01, iters=10):
        self.m, self.n = X.shape
        self.w = np.random(self.n)
        self.b = np.random()
        self.w, self.b = self.gradient_descent(X, y, alpha, iters)