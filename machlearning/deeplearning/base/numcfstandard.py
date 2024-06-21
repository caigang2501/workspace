import math
import numpy as np
from keras import datasets
import matplotlib.pyplot as plt

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = x.reshape(60000,784)
x = 2*x/255 - 1
x_val = x_val.reshape(10000,784)
print(x.shape,x_val.shape)


class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, 
        bias=None):
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation 
        self.last_activation = None 
        self.error = None 
        self.delta = None 
    
    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias 
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        if self.activation is None:
            return r 
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r

class NeuralNetwork:
    def __init__(self):
        self._layers = [] 
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def backpropagation(self, X, y, learning_rate):
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))): 
            layer = self._layers[i] 
            if layer == self._layers[-1]: 
                layer.error = y - output 
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else: 
                next_layer = self._layers[i + 1] 
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        for i in range(len(self._layers)):
            layer = self._layers[i]
            o_i = np.atleast_2d(X if i == 0 else self._layers[i -1].last_activation)
            layer.weights += layer.delta * o_i.T * learning_rate
            layer.bias += layer.delta * learning_rate


        return list(output).index(max(output)) == list(y).index(1) 

    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        y_onehot = np.zeros((y_train.shape[0], 10))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []
        accuracy = 0
        for i in range(max_epochs): 
            for j in range(len(X_train)): 
                predict = self.backpropagation(X_train[j], y_onehot[j], learning_rate)
                if j > 59000 and predict:
                    accuracy += 1
            if i % 1 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
                # print('Accuracy: %.2f%%' % (self.accuracy(self.predict(X_test), 
                # y_test.flatten()) * 100))
                print(accuracy)
                accuracy = 0
        return mses

nn = NeuralNetwork() # 实例化网络类
nn.add_layer(Layer(784, 20, 'sigmoid')) 
nn.add_layer(Layer(20, 10, 'sigmoid')) 
# nn.add_layer(Layer(50, 25, 'sigmoid')) 
# nn.add_layer(Layer(25, 2, 'sigmoid')) 
nn.train(x, x_val, y, y_val,0.1,50)