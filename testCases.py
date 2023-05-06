import numpy as np

def initialize_parameters_test_case():
    n_x, n_h, n_y = 3, 2, 1
    return n_x, n_h, n_y

def initialize_parameters_deep_test_case():
    layer_dims = [5, 4, 3]
    return layer_dims

def linear_forward_test_case():
    A = np.array([[1.0, 2.0], [-1.0, 0.0]])
    W = np.array([[0.5, -1.0], [1.0, 0.0]])
    b = np.array([[1.0], [0.0]])
    return A, W, b

def linear_activation_forward_test_case():
    A_prev = np.array([[1.0, 2.0], [-1.0, 0.0]])
    W = np.array([[0.5, -1.0], [1.0, 0.0]])
    b = np.array([[1.0], [0.0]])
    return A_prev, W, b

def linear_model_forward_test_case():
    X = np.array([[1.0, 2.0], [-1.0, 0.0]])
    parameters = {'W1': np.array([[0.5, -1.0], [1.0, 0.0]]),
                  'b1': np.array([[1.0], [0.0]]),
                  'W2': np.array([[0.5, -1.0]]),
                  'b2': np.array([[1.0]])}
    return X, parameters

def compute_cost_test_case():
    AL = np.array([[0.5, 0.7]])
    Y = np.array([[1.0, 0.0]])
    return AL, Y

def linear_backward_test_case():
    dZ = np.array([[1.0, -1.0], [0.5, 0.0]])
    cache = (np.array([[1.0, 2.0], [-1.0, 0.0]]),
             np.array([[0.5, -1.0], [1.0, 0.0]]),
             np.array([[1.0], [0.0]]))
    return dZ, cache

def linear_activation_backward_test_case():
    dA = np.array([[1.0, -1.0], [0.5, 0.0]])
    cache = ((np.array([[1.0, 2.0], [-1.0, 0.0]]),
              np.array([[0.5, -1.0], [1.0, 0.0]]),
              np.array([[1.0], [0.0]])),
             np.array([[0.5, 0.5]]))
    return dA, cache

def L_model_backward_test_case():
    AL = np.array([[0.5, 0.7]])
    Y = np.array([[1.0, 0.0]])
    caches = [((np.array([[1.0, 2.0], [-1.0, 0.0]]),
                np.array([[0.5, -1.0], [1.0, 0.0]]),
                np.array([[1.0], [0.0]])),
               np.array([[0.5, 0.5]]))]
    return AL, Y,

def update_parameters_test_case():
    parameters = {'W1': np.array([[0.5, -1.0], [1.0, 0.0]]),
                  'b1': np.array([[1.0], [0.0]]),
                  'W2': np.array([[0.5, -1.0]]),
                  'b2': np.array([[1.0]])}
    grads = {'dW1': np.array([[0.1, 0.2], [-0.1, 0.0]]),
             'db1': np.array([[0.1], [0.0]]),
             'dW2': np.array([[0.1, 0.2]]),
             'db2': np.array([[0.1]])}
    learning_rate = 0.1
    return parameters, grads, learning_rate