import tensorlayer as tl
import tensorflow as tf
import pickle
import numpy as np
import logging

N_GRAM = 1

# Size of vocabulary; less frequent words will be treated as "unknown"
# VOCAB_SIZE = 100000
VOCAB_SIZE = 64189
N_BUCKETS = 3

# Size of the embedding vectors
EMBEDDING_SIZE = 300

# Number of epochs for which the model is trained
N_EPOCH = 3

# Size of training mini-batches
BATCH_SIZE = 32
# SEQ_LENGTH = 50


def sort_and_split(sequences, targets):
    new_seqs = []
    new_tars = []
    seq_buckets = {'50': {'seqs':[], 'tars':[]},
                   '100': {'seqs':[], 'tars':[]},
                   '300': {'seqs':[], 'tars':[]},
                   '500': {'seqs':[], 'tars':[]},
                   '1000': {'seqs':[], 'tars':[]},
                   '1500': {'seqs':[], 'tars':[]}}
    seq_coll = []
    for seq, tar in zip(*(sequences, targets)):
        length = len(seq)
        if length <= 100:
            seq_buckets['100']['seqs'].append(seq)
            seq_buckets['100']['tars'].append(tar)
        elif length <= 300:
            seq_buckets['300']['seqs'].append(seq)
            seq_buckets['300']['tars'].append(tar)
        elif length <= 500:
            seq_buckets['500']['seqs'].append(seq)
            seq_buckets['500']['tars'].append(tar)
        elif length <= 1000:
            seq_buckets['1000']['seqs'].append(seq)
            seq_buckets['1000']['tars'].append(tar)
        else:
            seq_buckets['1500']['seqs'].append(seq)
            seq_buckets['1500']['tars'].append(tar)
    for key in seq_buckets:
        seq_buckets[key]['seqs'] = np.array(tl.prepro.pad_sequences(seq_buckets[key]['seqs'], maxlen=int(key), value=64191))
        seq_buckets[key]['tars'] = np.array(seq_buckets[key]['tars'])
        for index, seq in enumerate(seq_buckets[key]['seqs']):
            for i in range(int(key) // 100):
                new_seqs.append(seq[i*100:(i+1)*100])
                new_tars.append(seq_buckets[key]['tars'][index])
    return new_seqs, new_tars

def load_and_preprocess_imdb_data():
    t_pkl = open('D://Codes/NSE/data/output/train_data.pkl', 'rb')
    train = pickle.load(t_pkl)
    v_pkl = open('D://Codes/NSE/data/output/valid_data.pkl', 'rb')
    valid = pickle.load(v_pkl)
    X_train = []
    y_train = []
    for item in train:
        X_train.append(list(item['content']))
        y_train.append(1 if item['label'] is False else 0)
    X_valid = []
    y_valid = []
    for item in valid:
        X_valid.append(list(item['content']))
        y_valid.append(1 if item['label'] is False else 0)
    return X_train, y_train, X_valid, y_valid

def load_and_preprocess_test_imdb_data():
    t_pkl = open('D://Codes/NSE/data/output/test_data.pkl', 'rb')
    test = pickle.load(t_pkl)
    X_test = []
    y_test = []
    for item in test:
        X_test.append(list(item['content']))
        y_test.append(1 if item['label'] is False else 0)
    return X_test, y_test


# X_train, y_train, X_valid, y_valid = load_and_preprocess_imdb_data()
# X_train, y_train = sort_and_split(X_train, y_train)
# X_valid, y_valid = sort_and_split(X_valid, y_valid)
#
# train = {'seqs': X_train, 'tars': y_train}
# valid = {'seqs': X_valid, 'tars': y_valid}
# t_p = open('D://Codes/NSE/data/output/splited_train_data.pkl', 'wb')
# pickle.dump(train, t_p)
# v_p = open('D://Codes/NSE/data/output/splited_valid_data.pkl', 'wb')
# pickle.dump(valid, v_p)


# X_test, y_test = load_and_preprocess_test_imdb_data()
# X_test, y_test = sort_and_split(X_test, y_test)
# test = {'seqs': X_test, 'tars': y_test}
# t_p = open('D://Codes/NSE/data/output/splited_test_data.pkl', 'wb')
# pickle.dump(test, t_p)


t_p = open('D://Codes/NSE/data/output/splited_test_data.pkl', 'rb')
test = pickle.load(t_p)
print(np.shape(test['seqs']))
for seq in test['seqs']:
    print(seq)




