import pickle
import numpy as np
import tensorlayer as tl


t_pkl = open('D://Codes/NSE/data/output/train_data.pkl', 'rb')
train = pickle.load(t_pkl)
print(train[0])
v_pkl = open('D://Codes/NSE/data/output/valid_data.pkl', 'rb')
valid = pickle.load(v_pkl)
print(np.shape(valid))


# VOCAB_SIZE = 100000
#
X_train, y_train, X_test, y_test = tl.files.load_imdb_dataset(nb_words=10000)

print(X_train)
print(y_train)
# for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size=30, shuffle=True):
#     print(np.shape(X_batch))
#     print(np.shape(y_batch))