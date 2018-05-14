import numpy as np
import tensorflow as tf
from scipy import stats

import input_layer

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('TRAIN_CLASS', 'train_classification.tfrecords', '')
flags.DEFINE_string('TEST_CLASS', 'test_classification.tfrecords', '')
flags.DEFINE_string('VALID_CLASS', 'validate_classification.tfrecords', '')

flags.DEFINE_string('input_dir', 'D://Codes/NSE/data/output', 'Path to the output folder.')
flags.DEFINE_string('vocab_file', 'D://Codes/NSE/data/output/vocab_unigram.txt', 'Path to the vocabulary file.')
flags.DEFINE_string('vocab_freqs_file', 'D://Codes/NSE/data/output/vocab_unigram_freq.txt', 'Path to the vocabulary frequence file.')
flags.DEFINE_string('simlex_999', 'D://Codes/NSE/data/raw/embeddingTest/simlex999.txt', 'Path to the Hill et al.s(2014) SimeLex-999.')

flags.DEFINE_integer('num_classes', 2, 'Number of classes for classification')
flags.DEFINE_integer('batch_size', 1, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 100, 'Number of timesteps for BPTT')
flags.DEFINE_integer('vocab_size', 71458, '')
flags.DEFINE_integer('embedding_size', 50, '')
flags.DEFINE_integer('window_size', 3, '')
flags.DEFINE_integer('hidden_size', 20, '')
flags.DEFINE_float('embedding_keep_prob', 1.0, '')
flags.DEFINE_float('linear_keep_prob', 1.0, '')
flags.DEFINE_float('loss_mix_weight', 0.5, '')
flags.DEFINE_boolean('embedding_normalized', False, '')



class SSWE_u(object):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 window_size,
                 hidden_size,
                 vocab_freqs,
                 embedding_keep_prob=1.0,
                 linear_keep_prob=1.0,
                 normalized=False,
                 loss_mix_weight=0.5):
        self.layers = {}
        self.output = {}
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.normalized = normalized
        self.vocab_freqs = vocab_freqs
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.embedding_keep_prob = embedding_keep_prob
        self.linear_keep_prob = linear_keep_prob
        self.loss_mix_weight = loss_mix_weight

        emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embedding_size], minval=-1.0, maxval=1.0), name='embedding')
        if self.normalized:
            assert self.vocab_freqs is not None
            emb = self._normalize(emb, self.vocab_size, self.vocab_freqs)
        self.layers['embedding'] = emb

        w_1 = tf.Variable(tf.random_uniform(shape=[self.embedding_size*self.window_size, self.hidden_size], minval=-1.0, maxval=1.0))
        b_1 = tf.Variable(tf.zeros([self.hidden_size]))
        self.layers['W_1'] = w_1
        self.layers['b_1'] = b_1

        w_2 = tf.Variable(tf.random_uniform(shape=[self.hidden_size, 2], minval=-1.0, maxval=1.0))
        b_2 = tf.Variable(tf.zeros([2]))
        self.layers['W_2'] = w_2
        self.layers['b_2'] = b_2


    def _normalize(self, emb, vocab_size, vocab_freqs):
        self.vocab_freqs = tf.constant(vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keepdims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keepdims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev


    # each x is a trigram whose element is also a n-gram
    # shape(x) = [batch_size, 3]
    def __call__(self, x, label):
        embedded = tf.nn.embedding_lookup(self.layers['embedding'], x)
