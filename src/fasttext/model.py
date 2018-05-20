#!/usr/bin/env python
"""
This demo implements FastText[1] for sentence classification.
FastText is a simple model for text classification with performance often close
to state-of-the-art, and is useful as a solid baseline.
There are some important differences between this implementation and what
is described in the paper. Instead of Hogwild! SGD[2], we use Adam optimizer
with mini-batches. Hierarchical softmax is also not supported; if you have
a large label space, consider utilizing candidate sampling methods provided
by TensorFlow[3].
After 5 epochs, you should get test accuracy close to 90.9%.
[1] Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016).
    Bag of Tricks for Efficient Text Classification.
    http://arxiv.org/abs/1607.01759
[2] Recht, B., Re, C., Wright, S., & Niu, F. (2011).
    Hogwild: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
    In Advances in Neural Information Processing Systems 24 (pp. 693–701).
[3] https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling
"""

import array
import hashlib
import time
import os
import pickle
import numpy as np
import tensorflow as tf

import tensorlayer as tl

# Hashed n-grams with 1 < n <= N_GRAM are included as features
# in addition to unigrams.
N_GRAM = 1

# Size of vocabulary; less frequent words will be treated as "unknown"
# VOCAB_SIZE = 100000
VOCAB_SIZE = 64191

# Number of buckets used for hashing n-grams
N_BUCKETS = 100000

# Size of the embedding vectors
EMBEDDING_SIZE = 300

# Number of epochs for which the model is trained
N_EPOCH = 3

# Size of training mini-batches
BATCH_SIZE = 32

# Path to which to save the trained model
MODEL_FILE_PATH = 'save/demo/'


class FastTextClassifier(object):
    """Simple wrapper class for creating the graph of FastText classifier."""

    def __init__(self, vocab_size, embedding_size, n_labels):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_labels = n_labels

        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        # Network structure
        network = tl.layers.AverageEmbeddingInputlayer(self.inputs, self.vocab_size, self.embedding_size)
        self.network = tl.layers.DenseLayer(network, self.n_labels)

        # Training operation
        cost = tl.cost.cross_entropy(self.network.outputs, self.labels, name='cost')
        self.train_op = tf.train.AdamOptimizer().minimize(cost)

        # Predictions
        self.prediction_probs = tf.nn.softmax(self.network.outputs)
        self.predictions = tf.argmax(self.network.outputs, axis=1, output_type=tf.int32)
        # self.predictions = tf.cast(tf.argmax(             # for TF < 1.2
        #     self.network.outputs, axis=1), tf.int32)

        # Evaluation
        are_predictions_correct = tf.equal(self.predictions, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(are_predictions_correct, tf.float32))

    def save(self, sess, filename):
        tl.files.save_npz(self.network.all_params, name=filename, sess=sess)

    def load(self, sess, filename):
        tl.files.load_and_assign_npz(sess, name=filename, network=self.network)


def augment_with_ngrams(unigrams, unigram_vocab_size, n_buckets, n=2):
    """Augment unigram features with hashed n-gram features."""

    def get_ngrams(n):
        return list(zip(*[unigrams[i:] for i in range(n)]))

    def hash_ngram(ngram):
        bytes_ = array.array('L', ngram).tobytes()
        hash_ = int(hashlib.sha256(bytes_).hexdigest(), 16)
        return unigram_vocab_size + hash_ % n_buckets

    return unigrams + [hash_ngram(ngram) for i in range(2, n + 1) for ngram in get_ngrams(i)]


# def load_and_preprocess_imdb_data(n_gram=None):
#     """Load IMDb data and augment with hashed n-gram features."""
#     X_train, y_train, X_test, y_test = tl.files.load_imdb_dataset(nb_words=VOCAB_SIZE)
#
#     if n_gram is not None:
#         X_train = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_train])
#         X_test = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_test])
#
#     return X_train, y_train, X_test, y_test

def load_and_preprocess_imdb_data(n_gram=None):
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
    if n_gram is not None:
        X_train = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_train])
        X_valid = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_valid])
    return X_train, y_train, X_valid, y_valid


def load_and_preprocess_imdb_test_data(n_gram=None):
    t_pkl = open('D://Codes/NSE/data/output/test_data.pkl', 'rb')
    test = pickle.load(t_pkl)
    X_test = []
    y_test = []
    for item in test:
        X_test.append(list(item['content']))
        y_test.append(1 if item['label'] is False else 0)
    if n_gram is not None:
        X_test = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_test])
    return X_test, y_test


def train_valid_and_save_model():
    X_train, y_train, X_valid, y_valid = load_and_preprocess_imdb_data(N_GRAM)
    classifier = FastTextClassifier(
        # vocab_size=VOCAB_SIZE + N_BUCKETS,
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        n_labels=2,
    )

    with tf.Session() as sess:
        tl.layers.initialize_global_variables(sess)

        i = 0
        for epoch in range(N_EPOCH):
            start_time = time.time()
            print('Epoch %d/%d' % (epoch + 1, N_EPOCH))
            for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True):
                i += 1
                sess.run(
                    classifier.train_op, feed_dict={
                        classifier.inputs: tl.prepro.pad_sequences(X_batch),
                        classifier.labels: y_batch,
                    }
                )

                if i % 100 == 0:
                    print("took %.5fs" % (time.time() - start_time))

                    valid_accuracy = sess.run(
                        classifier.accuracy, feed_dict={
                            classifier.inputs: tl.prepro.pad_sequences(X_valid),
                            classifier.labels: y_valid,
                        }
                    )
                    print('Valid accuracy: %.5f' % valid_accuracy)
                    classifier.save(sess, MODEL_FILE_PATH + 'model_' + str(i) + '.npz')
            classifier.save(sess, MODEL_FILE_PATH + 'model_' + str(epoch) + '.npz')
        classifier.save(sess, MODEL_FILE_PATH + 'model_' + str(i) + '.npz')


def load_and_test_model():
    X_test, y_test = load_and_preprocess_imdb_test_data(N_GRAM)
    classifier = FastTextClassifier(
        # vocab_size=VOCAB_SIZE + N_BUCKETS,
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        n_labels=2,
    )

    with tf.Session() as sess:
        foldername = MODEL_FILE_PATH
        pathDir = os.listdir(foldername)
        for filename in pathDir:
            fullname = os.path.join('%s/%s' % (foldername, filename))
            print(fullname)
            classifier.load(sess, fullname)

            start_time = time.time()
            test_accuracy = []

            for X_batch, y_batch in tl.iterate.minibatches(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False):
                acc = sess.run(
                    classifier.accuracy, feed_dict={
                        classifier.inputs: tl.prepro.pad_sequences(X_batch),
                        classifier.labels: y_batch,
                        # classifier.embedding.pretrained_embeddings: pretrained_wv
                    })
                test_accuracy.append(acc)

            test_accuracy = np.mean(test_accuracy)
            print('Test accuracy: %.5f' % test_accuracy)
            print("took %.5fs" % (time.time() - start_time))




def load_and_test_distilled_model():
    X_test, y_test = load_and_preprocess_imdb_test_data(N_GRAM)
    classifier = FastTextClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        n_labels=2,
    )
    with tf.Session() as sess:
        classifier.load(sess, 'D://Codes/NSE/src/fasttext/save/uninited_unigram/model_1.npz')
        distilled_vecs = np.load('D://Codes/NSE/src/fasttext/save/uninited_unigram/model_1_distilled.npy')
        tl.files.assign_params(sess, [distilled_vecs], classifier.network)

        start_time = time.time()
        test_accuracy = []

        for X_batch, y_batch in tl.iterate.minibatches(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False):
            acc = sess.run(
                classifier.accuracy, feed_dict={
                    classifier.inputs: tl.prepro.pad_sequences(X_batch, value=0),
                    # classifier.inputs: tl.prepro.pad_sequences(X_batch, value=64189),
                    classifier.labels: y_batch,
                    # classifier.embedding.pretrained_embeddings: pretrained_wv
                })
            test_accuracy.append(acc)

        test_accuracy = np.mean(test_accuracy)
        print('Test accuracy: %.5f' % test_accuracy)
        print("took %.5fs" % (time.time() - start_time))


if __name__ == '__main__':
    # with tf.device("/cpu:0"):
        # train_valid_and_save_model()
        # load_and_test_model()
    load_and_test_distilled_model()