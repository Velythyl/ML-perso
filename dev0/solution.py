import numpy as np


def make_array_from_list(some_list):
    return np.array(some_list)


def make_array_from_number(num):
    return np.array([num])


class NumpyBasics:
    def add_arrays(self, a, b):
        return np.add(a, b)

    def add_array_number(self, a, num):
        return a + num

    def multiply_elementwise_arrays(self, a, b):
        return np.multiply(a, b)

    def dot_product_arrays(self, a, b):
        return np.dot(a, b)

    def dot_1d_array_2d_array(self, a, m):
        # consider the 2d array to be like a matrix
        return a @ m


def test():
    x = NumpyBasics()
    print(make_array_from_list([1, 2, 3]))
    print(make_array_from_number(4))
    print(x.add_arrays(np.array([1, 1, 1]), np.array([1, 1, 1])))
    print(x.add_array_number(np.array([1, 1, 1]), 6))
    print(x.multiply_elementwise_arrays(np.array([1, 2, 3]), np.array([1, 2, 3])))
    print(x.dot_product_arrays(np.array([1, 2, 3]), np.array([1, 2, 3])))
    print(x.dot_1d_array_2d_array(np.array([1, 2, 3]), np.array([[5, 1, 3], [1, 1, 1], [1, 2, 1]])))
