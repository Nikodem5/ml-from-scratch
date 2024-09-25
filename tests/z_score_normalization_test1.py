import numpy as np
from src.utils.z_score_normalization import ZScoreScaler
from sklearn.preprocessing import StandardScaler

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

#TODO add timing
my_scaler = ZScoreScaler()
sklearn_scaler = StandardScaler()

my_data = my_scaler.fit_transform(X)
sklearn_data = sklearn_scaler.fit_transform(X)

print(f"Original data: \n{X}")
print(f"My scaled data: \n{my_data}")
print(f"sklearn's scaled data: \n{sklearn_data}")
