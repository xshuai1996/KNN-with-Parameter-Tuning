import numpy as np

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    def train(self, features, labels):
        """
        Loading of training data.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.features = features
        self.labels = labels

    def predict(self, features):
        """
        :param features: List[List[float]]
        :return: List[int]
        """
        pred = []
        for i in features:
            neighbors_labels = self.get_k_neighbors(i)
            pred.append(0 if neighbors_labels.count(0) >= neighbors_labels.count(1) else 1)
        return pred

    def get_k_neighbors(self, point):
        """
        :param point: List[float]
        :return:  List[int]
        """
        distances = sorted([[self.distance_function(point, self.features[i]), self.labels[i]] for i in range(len(self.labels))], key=lambda x: x[0])
        return [i[1] for i in distances[:min(self.k, len(self.labels))]]

if __name__ == '__main__':
    print(np.__version__)
