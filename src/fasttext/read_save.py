import tensorlayer as tl
import numpy as np
import pickle


# load_params = tl.files.load_npz(name='save/inited/model_0.npz')
#
# print(len(load_params))
# print(np.shape(load_params[2]))


def load_and_preprocess_imdb_test_data(n_gram=None):
    t_pkl = open('D://Codes/NSE/data/output/test_data.pkl', 'rb')
    test = pickle.load(t_pkl)
    X_test = []
    y_test = []
    for item in test:
        X_test.append(list(item['content']))
        y_test.append(1 if item['label'] is False else 0)
    return X_test, y_test


X_test, y_test = load_and_preprocess_imdb_test_data()
for x, y in zip(X_test, y_test):
    print(x, y)