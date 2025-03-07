X = [[1.6], [1.6], [2],[2.2],[2.3], [3],[4],[4]]
y = [0, 0, 1,1,1, 2,3,3]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
KNeighborsClassifier(...)
print(neigh.predict([[2.1]]))


