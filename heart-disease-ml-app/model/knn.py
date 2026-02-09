
from sklearn.neighbors import KNeighborsClassifier

def build_model():
    return KNeighborsClassifier(n_neighbors=5)
