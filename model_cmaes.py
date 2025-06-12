import numpy as np

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    x = np.array(x) - np.min(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def unpack_weights(vec):
    lin_1 = np.reshape(vec[:900], newshape=(30, 30))
    lin_2 = np.reshape(vec[900:1860], newshape=(32, 30))

    return lin_1, lin_2

class Model():
    def __call__(self, x, vec):
        A, B = unpack_weights(vec)

        x = relu(np.dot(A, x))
        x = np.dot(B, x)

        return softmax(x)
