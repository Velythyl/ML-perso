import math

import numpy as np

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
    def only_keep_label_1(self, iris):
        return iris[np.where(iris[:,4] == 1)]

    def feature_means(self, iris):
        return np.mean(strip_labels(iris), axis=0)

    def covariance_matrix(self, iris):
        return np.cov(strip_labels(iris), rowvar=False) # TODO

    def feature_means_class_1(self, iris):
        return self.feature_means(self.only_keep_label_1(iris))

    def covariance_matrix_class_1(self, iris):
        return self.covariance_matrix(self.only_keep_label_1(iris))

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

        return np.argmax(np.sum(rbfs, axis=0))+1   # doit ajouter 1 car index partent de 0

    def compute_predictions(self, test_data):
        return np.array([self.predict_vector(v) for v in test_data])


def split_dataset(iris):
    train_set = []
    validation_set = []
    test_set = []

    for i in range(len(iris)):
        mod = i % 5
        ele = iris[i]

        if mod == 3:
            validation_set.append(ele)
        elif mod == 4:
            test_set.append(ele)
        else:
            train_set.append(ele)


    return (np.array(train_set), np.array(validation_set), np.array(test_set))



class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def error_rate(self, p):
        p.train(self.x_train, self.y_train)
        pred = p.compute_predictions(self.x_val)

        nb_bad = 0
        for i in range(len(pred)):
            nb_bad += int(int(pred[i]) != int(self.y_val[i]))

        return nb_bad / len(pred)

    def hard_parzen(self, h):
        return self.error_rate(HardParzen(h))

    def soft_parzen(self, sigma):
        return self.error_rate(SoftRBFParzen(sigma))

# Question 5
def q5(plot=False):
    iris = np.loadtxt("iris.txt")

    if plot:
        import matplotlib.pyplot as plt

    split = split_dataset(iris)
    train = split[0]
    valid = split[1]
    test = split[2]

    err = ErrorRate(strip_labels(train), train[:,-1], strip_labels(valid), valid[:,-1])

    def str_cap(n):
        return str(n)[:3]

    def annotate(x, y, color):
        for a,b in zip(x, y):
            plt.text(a, b, "    ("+str_cap(a)+", "+str_cap(b)+")", color=color)

    # Best = 1
    h_list = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    vh_list = [err.hard_parzen(h) for h in h_list]
    if plot:
        plt.plot(h_list, vh_list, label="Hard Parzen", marker=".", color="b")
        annotate(h_list, vh_list, "b")

    # Best = 0.3
    s_list = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    vs_list = [err.soft_parzen(s) for s in s_list]
    if plot:
        plt.plot(s_list, vs_list, label="Soft Parzen", marker="v", color="g")
        annotate(s_list, vs_list, "g")
        plt.ylabel("Error rate")
        plt.xlabel("Value of σ and h")
        plt.legend()
        plt.show()

    return np.array([h_list[np.argmin(vh_list)], s_list[np.argmin(vs_list)]])

def get_test_errors(iris):
    split = split_dataset(iris)
    train = split[0]
    valid = split[1]
    test = split[2]

    err = ErrorRate(strip_labels(train), train[:,-1], strip_labels(test), test[:,-1])

    hstar, sstar = q5()

    return np.array([err.hard_parzen(hstar), err.soft_parzen(sstar)])

def random_projections(X, A):
    pass