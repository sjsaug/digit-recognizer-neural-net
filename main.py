import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/train.csv')
data.head()
# Extract data
data = np.array(data)
m, n = data.shape #rows, features
np.random.shuffle(data)

data_dev = data[0:1000].T
X_dev = data_dev[1:n]
Y_dev = data_dev[0]

data_train = data[1000:m].T
X_train = data_train[1:n]
Y_train = data_train[0]

# Def functions
def init_parameters():
    W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b2 = np.zeros((10, 1))
    return W1, W2, b1, b2

def LeakyReLU(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha) # fix ReLU

def softmax(x):
    x -= np.max(x, axis=0)  # subtract max value for numerical stability
    A = np.exp(x) / np.sum(np.exp(x), axis=0)
    return A
    
def forward_propagation(W1, W2, b1, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = LeakyReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, Z2, A1, A2
    
def one_hot(Y):
    OH_Y = np.zeros((Y.size, Y.max() + 1))
    OH_Y[np.arange(Y.size), Y] = 1
    OH_Y = OH_Y.T # from each row being example to each col being example; following math
    return OH_Y

def dx_LeakyReLU(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha) # fix dx_ReLU
    
def backwards_propagation(Z1, Z2, A1, A2, W2, X, Y):
    m = Y.size
    OH_Y = one_hot(Y)
    dZ2 = A2 - OH_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * dx_LeakyReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, dW2, db1, db2

def update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, a): # a = alpha
    W1 = W1 - a * dW1
    W2 = W2 - a * dW2
    b1 = b1 - a * db1
    b2 = b2 - a * db2
    return W1, W2, b1, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(prediction, Y):
    print(prediction, Y)
    return np.sum(prediction == Y) / Y.size

def gradient_descent(X, Y, iter, a):
    W1, W2, b1, b2 = init_parameters()
    for i in range(iter): # num of iterations
        Z1, Z2, A1, A2 = forward_propagation(W1, W2, b1, b2, X)
        dW1, dW2, db1, db2 = backwards_propagation(Z1, Z2, A1, A2, W2, X, Y)
        W1, W2, b1, b2 = update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, a)
        if i % 10 == 0: # every 10th iteration
            print(f"Iteration : {i}")
            print(f"Accuracy : {get_accuracy(get_predictions(A2), Y)}")
    return W1, W2, b1, b2

# Run
iterations = 5000
alpha = 0.001
W1, W2, b1, b2 = gradient_descent(X_train, Y_train, iterations, alpha) # way better results with lower alpha & higher iterations

# Test
def make_predictions(X, W1, W2, b1, b2):
    _, _, _, A2 = forward_propagation(W1, W2, b1, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, W2, b1, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, W2, b1, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, W2, b1, b2)
test_prediction(1, W1, W2, b1, b2)
test_prediction(2, W1, W2, b1, b2)
test_prediction(3, W1, W2, b1, b2)