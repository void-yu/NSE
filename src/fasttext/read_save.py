import tensorlayer as tl
import numpy as np


load_params = tl.files.load_npz(name='save/inited/model_0.npz')

print(len(load_params))
print(np.shape(load_params[2]))

