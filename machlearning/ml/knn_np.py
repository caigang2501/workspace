import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# 示例用法
X_train = np.array([[1,2,3], [2,3,4], [3,4,5], [5,6,7]])
y_train = np.array([0, 0, 1, 1])

knn = KNN(k=2)
knn.fit(X_train, y_train)

X_test = np.array([[4,5,6], [1,1,1]])
predictions = knn.predict(X_test)
print(predictions)  # 输出：[1 0]
