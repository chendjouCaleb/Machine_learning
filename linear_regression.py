import numpy as np
class LinearRegression:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.coeff = np.array([])

    def train(self):
        #print(self.X)
        X = np.column_stack([self.features.T, np.ones(len(self.features))])
        Y = self.targets
        X = np.matrix(X)
        Y = np.matrix(Y).T
        W = X.T * X
        W = W.I
        W = W * X.T
        W = W * Y
        self.coeff = W
        print(self.coeff)


class SequentialLinearRegression:
    def __init__(self, features, targets, learning_rate=0.1, epoch_num=100):
        self.features = features
        self.targets = targets
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.coeff = np.array([])

    def train(self):
        #print(self.X)
        lr = self.learning_rate
        X = np.column_stack([self.features.T, np.ones(len(self.features))])
        Y = self.targets
        W = np.zeros(X.shape[1])
        for epoch in range(self.epoch_num):
            for i in range(len(X)):
                yt = np.sum(W * X[i])
                W += lr * (Y[i] - yt) * X[i]
            #print(W)
        self.coeff = W
        print(self.coeff)
