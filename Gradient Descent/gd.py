import numpy as np
import sys
from matplotlib import pyplot as plt

def gradient_descent(iter, alpha, epsilon):
    x = np.random.normal(loc = 0, scale = 3, size=(1, 2))
    path = np.array(x)
    i = 0
    flag = True
    while i < iter and flag:
        i += 1
        gradient = np.array([[
            2 * x[0, 0] + x[0, 1],
            2 * x[0, 1] + x[0,0]
        ]])
        x = x - alpha * gradient
        path = np.append(path, x, axis=0)
        if abs(np.sum(gradient, axis = 1)) < epsilon:
            flag = False
    return path

def power(x):
    return pow(x, 2)

class GD():

    def __init__(self):
        pass

    def _power(self, x):
        return pow(x, 2)

    def set_data(self, X, y):
        self.X = X
        self.N = X.shape[0]
        self.X = np.c_[np.ones(N), X]
        self.K = self.X.shape[1]
        self.y = y

    def optimize(self, ITER, alpha, epsilon):
        self.b = np.random.normal(loc = 0, scale = 0.3, size=(self.K, 1))
        self.loss = []
        i = 0
        flag = True
        while i < ITER and flag:
            i = i + 1
            for k in range(self.K):
                pd = 1/self.N * np.dot((np.matmul(self.X, self.b) - self.y)[:,0], self.X[:,k])
                self.b[k] = self.b[k] - alpha * pd
            loss = 0.5/self.N * np.sum(np.apply_along_axis(self._power, 1, np.matmul(self.X, self.b) - self.y))
            self.loss.append(loss)
            if loss < epsilon:
                flag = False

    def get_weights(self):
        return self.b

    def get_loss(self):
        return self.loss

if __name__ == '__main__':

    # input
    N = int(sys.argv[1])
    K = int(sys.argv[2])

    # generate X, Y, b
    X = np.c_[np.ones(N), np.random.normal(4, 3, size=(N, K - 1))]
    b = np.random.normal(0, 3, size=(K, 1))
    y = np.matmul(X, b)

    # optimize
    optimizer = GD()
    optimizer.set_data(X = X[:,1:], y = y)
    optimizer.optimize(ITER=2000, alpha = 0.01, epsilon = 0.1)
    loss = optimizer.get_loss()
    plt.plot(range(len(loss)), loss)
    plt.show()