import numpy as np

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    x = np.array(x) - np.min(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def unpack_weights(vec):
    lin_1 = np.reshape(vec[:144], newshape=(12, 12))
    lin_2 = np.reshape(vec[144:288], newshape=(12, 12))
    lin_3 = np.reshape(vec[288:], newshape=(18, 12))

    return lin_1, lin_2, lin_3

class Model():
    def __call__(self, x, vec):
        A, B, C = unpack_weights(vec)

        x = relu(np.dot(A, x))
        x = relu(np.dot(B, x))
        x = np.dot(C, x)
        x[:2] = softmax(x[:2])
        x[2:10] = softmax(x[2:10])
        x[10:] = softmax(x[10:])

        return x
