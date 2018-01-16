import numpy as np

from linear_regression import LinearRegression
from linear_regression import SequentialLinearRegression
from neural_network import MLP
features = np.linspace(0, 10, 11)
targets = 2 * features + 1 + np.random.normal(0, 1, 11)
lrs = SequentialLinearRegression(features, targets, epoch_num=2000, learning_rate=.01)
lr = LinearRegression(features, targets)

lrs.train()
lr.train()

print("LRS: {}".format(lrs.coeff))
print("LR: {}".format(lr.coeff))