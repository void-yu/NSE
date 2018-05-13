import pandas as pd
import numpy as np
import csv
import os
import tensorflow as tf
from scipy import stats
from sklearn.decomposition import PCA

import we

flags = tf.app.flags

def get_simlex999(vocab_filename):
    path = 'D://Codes/NSE/data/raw/embeddingTest/simlex999.txt'
    if not tf.gfile.Exists(path):
        raise ValueError('Failed to find file: %s' % path)
    else:
        with tf.gfile.Open(path) as f:
            evals = np.array(pd.read_csv(f, comment='#', sep='\t'))
            vocab = make_vocab_ids(vocab_filename)
            evals = [pair for pair in evals if pair[0] in vocab and pair[1] in vocab]
            print('Get %s pairs from simlex-999' % np.shape(evals)[0])
            return np.array(evals)


def make_vocab_ids(vocab_filename):
    with open(vocab_filename) as vocab_f:
        return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])



def eval_w2v_wordsim999():
    sswe = we.WordEmbedding('D://Codes/NSE/data/raw/embeddings/sswe-r.txt')
    simlex = get_simlex999('D://Codes/NSE/data/output/vocab_unigram.txt')
    results = []

    for pair in simlex:
        w1 = pair[0]
        w2 = pair[1]
        if w1 not in sswe.words or w2 not in sswe.words:
            print('Cant find', w1, w2)
            continue

        w1v = sswe.vecs[sswe.index[w1]]
        w2v = sswe.vecs[sswe.index[w2]]
        results.append([w1, w2, pair[2], np.sum(np.multiply(w1v, w2v))])

    print(stats.pearsonr([i[2] for i in results], [i[3] for i in results])[0], stats.spearmanr([i[2] for i in results], [i[3] for i in results]).correlation)

    for i in sorted(results, key=lambda x: x[3]):
        print(i)


def eval_w2v_principal_component():
    sswe = we.WordEmbedding('D://Codes/NSE/data/raw/embeddings/sswe-r.txt')
    wordpair = [line.strip().split() for line in open('D://Codes/NSE/data/used/seeds/wordpairs-greater-than-0.5').readlines()]
    sub_words_p = [line[0] for line in wordpair]
    sub_words_n = [line[1] for line in wordpair]
    matrix = []

    for index in range(len(sub_words_p)):
        w1 = sub_words_p[index]
        w2 = sub_words_n[index]
        if w1 not in sswe.words or w2 not in sswe.words:
            print('Cant find', w1, w2)
            continue
        w1v = sswe.vecs[sswe.index[w1]]
        w2v = sswe.vecs[sswe.index[w2]]
        center = (w1v + w2v) / 2
        matrix.append(w1v - center)
        matrix.append(w2v - center)
    pca = PCA(n_components=10)
    pca.fit(matrix)
    print(pca.explained_variance_ratio_)

eval_w2v_principal_component()