import numpy as np

class k_nearest_neighbors():
    def __init__(self):
        self.m = None
        self.n = None

    def euclidean_distance(self, p, q):
        sum_ = np. sum(np.square(p - q))
        dist = np.sqrt(sum_) 
        return dist
    
if __name__ == '__main__':
    knn = k_nearest_neighbors()

    p1 =np.array([4, 1])
    p2 = np.array([9, 6])

    print(knn.euclidean_distance(p1, p2)) # 5 * sqrt(2)
    print(np.multiply(5, np.sqrt(2)))

    q1 = np.array([5, 5, 5, 5])
    q2 = np.array([9, 9, 12, 17])

    print(knn.euclidean_distance(q1, q2)) # 15