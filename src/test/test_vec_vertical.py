import numpy as np
import tensorlayer as tl


# vec = np.load('vec_example.npy')
# dire = np.load('dire_example.npy')
vec = np.array([2., 2.])
dire = np.array([1., 0.])
print(np.linalg.norm(vec))
print(np.linalg.norm(dire))
vec_senti = np.sum(np.multiply(vec, dire)) / (np.linalg.norm(dire) * np.linalg.norm(dire)) * dire
print(vec_senti, np.linalg.norm(vec_senti))
vec_vertical_to_senti = vec - vec_senti
print(np.linalg.norm(vec_vertical_to_senti))
print(np.linalg.norm(np.multiply(vec_senti, vec_vertical_to_senti)))
