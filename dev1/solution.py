import math

import numpy as np
iris = np.loadtxt("iris.txt")

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

### UTILITY FUNCTIONS ###

# Retourne la matrice en parametre sans sa derniere colone (pour iris, sans les labels)
def strip_labels(iris):
    return iris[:,:-1]

# Pris de la solution de la demo 2
def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

# Cree un dictionaire ayant une clef pour chaque element de l initialise a 0
def list_dict(l):
    d = {}
    for ele in l:
        d[ele] = 0
    return d

def max_key(dictionary):
    return max(dictionary, key=dictionary.get)

### END UTILITY FUNCTIONS ###

class Q1:

    # Retourne les lignes du dataset dont le label est 1
    def only_keep_label1(self, iris):
        return iris[np.where(iris[:,4] == 1)]

    def feature_means(self, iris):
        return np.mean(strip_labels(iris), axis=0)

    def covariance_matrix(self, iris):
        return np.cov(iris) # TODO

    def feature_means_class_1(self, iris):
        return self.feature_means(self.only_keep_label1(iris))

    def covariance_matrix_class_1(self, iris):
        pass

class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.nb_of_labels = len(self.label_list)

    def predict_vector(self, vector):
        distances = minkowski_mat(vector, self.train_inputs)

        neighbours_indices = np.where(distances < self.h)[0]
        if len(neighbours_indices) == 0:
            return draw_rand_label(vector, self.label_list)

        count_dict = list_dict(self.label_list)
        for i in neighbours_indices:
            count_dict[self.train_labels[i]] += 1

        return max_key(count_dict)

    def compute_predictions(self, test_data):
        return np.array([self.predict_vector(v) for v in test_data])


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs

        nb_of_labels = len(np.unique(train_labels))
        def get_one_hot(label):
            empty = [0 for i in range(nb_of_labels)]
            empty[int(label) - 1] = 1

            return np.array(empty)

        self.one_hot_train_labels = np.array([get_one_hot(label) for label in train_labels])

    def rbf(self, distance):
        d = len(self.train_inputs[0])

        bot = ((2*math.pi) ** d/2) * (self.sigma ** d)
        top = math.e ** ((-1/2) * ((distance ** 2) / (self.sigma ** 2)))
        return top/bot

    def predict_vector(self, vector):
        distances = minkowski_mat(vector, self.train_inputs)

        rbfs = np.array([self.rbf(distances[i])*self.one_hot_train_labels[i] for i in range(len(distances))])

        return np.argmax(sum(rbfs))+1   # doit ajouter 1 car index partent de 0

    def compute_predictions(self, test_data):
        return np.array([self.predict_vector(v) for v in test_data])


def split_dataset(iris):
    pass


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        pass

    def soft_parzen(self, sigma):
        pass


def get_test_errors(iris):
    pass


def random_projections(X, A):
    pass


### TESTS ###
q1 = Q1()
print(q1.feature_means(iris))
print(q1.covariance_matrix(iris))
print(q1.feature_means_class_1(iris))

hp = HardParzen(4)
hp.train(strip_labels(iris), iris[:,-1])
print(hp.compute_predictions(strip_labels(iris)))

sp = SoftRBFParzen(2)
sp.train(strip_labels(iris), iris[:,-1])
print(sp.compute_predictions(strip_labels(iris)))