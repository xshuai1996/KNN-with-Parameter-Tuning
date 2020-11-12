import numpy as np
from knn import KNN

def f1_score(real_labels, predicted_labels):
    """
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    confusion_matrix = [[0, 0], [0, 0]]
    for i in range(len(real_labels)):
        confusion_matrix[int(real_labels[i])][int(predicted_labels[i])] += 1
    F1 = 2 * confusion_matrix[1][1] / (len(real_labels) + confusion_matrix[1][1] - confusion_matrix[0][0])
    return F1

class Distances:
    @staticmethod
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = sum([abs(point1[i] - point2[i]) / (abs(point1[i]) + abs(point2[i])) if (abs(point1[i]) + abs(point2[i]))!= 0 else 0 for i in range(len(point1))])
        return d

    @staticmethod
    def minkowski_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = pow(sum([pow(abs(point1[i] - point2[i]), 3) for i in range(len(point1))]), 1/3)
        return d


    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = pow(sum([pow(point1[i]-point2[i], 2) for i in range(len(point1))]), 1/2)
        return d


    @staticmethod
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = sum([point1[i] * point2[i] for i in range(len(point1))])
        return d

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        d = sum([point1[i] * point2[i] for i in range(len(point1))])
        length1 = pow(sum([pow(point1[i], 2) for i in range(len(point1))]), 1 / 2)
        length2 = pow(sum([pow(point2[i], 2) for i in range(len(point2))]), 1 / 2)
        similarity = d / (length1 * length2)
        return 1- similarity

    @staticmethod
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        d = - np.exp(- 1/ 2 * sum([pow(point1[i]-point2[i], 2) for i in range(len(point1))]))
        return d

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        Try different distance function in class Distances, and find the best k.
        :param distance_funcs: dictionary of distance function
        :param x_train: List[List[int]]
        :param y_train: List[int]
        :param x_val:  List[List[int]]
        :param y_val: List[int]
        """
        record = {}
        for name in distance_funcs:
            record[name] = []
            for k in range(1, 30, 2):
                knn = KNN(k, distance_funcs[name])
                knn.train(x_train, y_train)
                prediction = knn.predict(x_val)
                f1 = f1_score(y_val, prediction)
                record[name].append(f1)
                del knn
        max_f1 = -float('inf')
        for name in ['canberra', 'minkowski', 'euclidean', 'gaussian', 'inner_prod', 'cosine_dist']:
            if max(record[name]) > max_f1:
                max_f1 = max(record[name])
                self.best_distance_function = name
        for ind in range(len(record[self.best_distance_function])):
            if record[self.best_distance_function][ind] == max_f1:
                self.best_k = ind * 2 + 1
                break
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        self.best_model.train(x_train, y_train)

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively
        :param distance_funcs: dictionary of distance funtions
        :param scaling_classes: dictionary of scalers to normalized data.
        :param x_train: List[List[int]]
        :param y_train: List[int]
        :param x_val: List[List[int]]
        :param y_val: List[int]
        """
        record = {}
        for scaler_name in scaling_classes:
            for func_name in distance_funcs:
                record[(scaler_name, func_name)] = []
                for k in range(1, 30, 2):
                    knn = KNN(k, distance_funcs[func_name])
                    scaler = scaling_classes[scaler_name]()
                    knn.train(scaler(x_train), y_train)
                    prediction = knn.predict(scaler(x_val))
                    f1 = f1_score(y_val, prediction)
                    record[(scaler_name, func_name)].append(f1)
                    del knn
                    del scaler
        max_f1 = -float('inf')
        for scaler_name in ['min_max_scale', 'normalize']:
            for func_name in ['canberra', 'minkowski', 'euclidean', 'gaussian', 'inner_prod', 'cosine_dist']:
                if max(record[(scaler_name, func_name)]) > max_f1:
                    max_f1 = max(record[(scaler_name, func_name)])
                    self.best_distance_function = func_name
                    self.best_scaler = scaler_name
        for ind in range(len(record[(self.best_scaler, self.best_distance_function)])):
            if record[(self.best_scaler, self.best_distance_function)][ind] == max_f1:
                self.best_k = ind * 2 + 1
                break
        self.best_model = KNN(self.best_k, distance_funcs[self.best_distance_function])
        scaler = scaling_classes[self.best_scaler]()
        self.best_model.train(scaler(x_train), y_train)

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features to unit vectors
        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norm = []
        for i in features:
            devide = pow(Distances.inner_product_distance(i, i), 1/2)
            if devide != 0:
                norm.append([j/devide for j in i])
            else:
                norm.append([0 for _ in i])
        return norm

class MinMaxScaler:
    def __init__(self):
        self.Min_Max = []

    def __call__(self, features):
        """
        :param features: List[List[float]]
        :return: List[List[float]]
        """
        if not self.Min_Max:
            for i in range(len(features[0])):
                extract = [j[i] for j in features]
                self.Min_Max.append([min(extract), max(extract)])
        scaled = []
        for i in range(len(features)):
            row = []
            for j in range(len(features[0])):
                if self.Min_Max[j][1]-self.Min_Max[j][0] != 0:
                    row.append((features[i][j]-self.Min_Max[j][0]) / (self.Min_Max[j][1]-self.Min_Max[j][0]))
                else:
                    row.append(0)
            scaled.append(row)
        return scaled


