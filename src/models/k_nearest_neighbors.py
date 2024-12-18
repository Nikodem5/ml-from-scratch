import numpy as np

class KNearestNeighbors():
    def __init__(self, k):
        self.X_train = None
        self.y_train = None
        self.k = k

    def euclidean_distance(self, p, q):
        sum_of_squares = np. sum(np.square(p - q))
        dist = np.sqrt(sum_of_squares) 
        return dist
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
        
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)
    
if __name__ == '__main__':
    knn = KNearestNeighbors(5)

    p1 =np.array([4, 1])
    p2 = np.array([9, 6])

    print(knn.euclidean_distance(p1, p2)) # 5 * sqrt(2)
    print(np.multiply(5, np.sqrt(2)))

    q1 = np.array([5, 5, 5, 5])
    q2 = np.array([9, 9, 12, 17])

    print(knn.euclidean_distance(q1, q2)) # 15