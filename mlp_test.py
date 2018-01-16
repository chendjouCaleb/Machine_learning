from neural_network.MLP import MLP
import numpy as np

features = np.array([[1, 1], [2, 1], [-1, -2], [-0.1, -0.4]])
print(len(features[0]))
target = np.array([1, 1, 0, 0])
mlp = MLP(epoch=1000, num_hidden_layers=[2,2,4], learning_rate=.1)




mlp.fit(features, target)
mlp.train()
