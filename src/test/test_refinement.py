import numpy as np

v0_0 = np.reshape(np.array([1, 2, 3]), [-1, 1])
v1_0 = np.reshape(np.array([2, 3, 4]), [-1, 1])
v2_0 = np.reshape(np.array([3, 4, 5]), [-1, 1])
V = np.concatenate([v0_0, v1_0, v2_0], axis=1).T
W = np.array([[0, 1, 1/2], [1, 0, 1], [1/2, 1, 0]])
print(V)
M = 1 * V + 1 * np.matmul(V, W.T)
b = 1 + 1 * np.matmul(W, np.ones([3, 3]))
print(M)
print(b)
print(M / b)
