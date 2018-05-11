import tensorflow as tf
import math


class EmbeddingLayer(object):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 normalize,
                 keep_prob,
                 vocab_freqs):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.normalized = normalize
        self.keep_prob = keep_prob
        self.vocab_freqs = vocab_freqs

    def __call__(self, x):
        self.x = x

        with tf.variable_scope('embedding') as vs:
            embeddings = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embedding_size], minval=-1.0, maxval=1.0), name='embedding')
            if self.normalized:
                assert self.vocab_freqs is not None
                embeddings = self._normalize(embeddings, self.vocab_size, self.vocab_freqs)
            embedded = tf.nn.embedding_lookup(embeddings, self.x)

            self.trainable_weights = vs.global_variables()

            if self.keep_prob < 1.:
                embedded = tf.nn.dropout(embedded, keep_prob=self.keep_prob)
            return embedded

    def _normalize(self, emb, vocab_size, vocab_freqs):
        self.vocab_freqs = tf.constant(vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keepdims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keepdims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev


class LSTMLayer(object):
    def __init__(self,
                 rnn_cell_size,
                 num_layers=1,
                 keep_prob=1.,
                 name='LSTM'):
        self.rnn_cell_size = rnn_cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, length, initial_state):
        self.x = x
        self.length = length
        self.initial_state = initial_state

        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            lstm_cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.BasicLSTMCell(
                    self.rnn_cell_size,
                    forget_bias=1.0,
                    reuse=tf.get_variable_scope().reuse,
                    state_is_tuple=True)
                for _ in range(self.num_layers)
            ])

            # shape(x) = (batch_size, num_timesteps, embedding_dim)
            # Convert into a time-major list for static_rnn
            self.x = tf.unstack(tf.transpose(self.x, perm=[1, 0, 2]))

            # lstm_out, next_state = tf.nn.dynamic_rnn(
            lstm_out, next_state = tf.nn.static_rnn(
                lstm_cell, self.x, initial_state=self.initial_state, sequence_length=self.length)

            # Merge time and batch dimensions
            # shape(lstm_out) = timesteps * (batch_size, cell_size)
            lstm_out = tf.concat(lstm_out, 0)
            # shape(lstm_out) = (timesteps*batch_size, cell_size)

            if self.keep_prob < 1.:
                lstm_out = tf.nn.dropout(lstm_out, self.keep_prob)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True

        return lstm_out, next_state


class LogitsLayer(object):
    def __init__(self,
                 input_size,
                 hidden_size,
                 keep_prob=1.,
                 num_classes=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.num_classes = num_classes

    def __call__(self, inputs):
        w_1 = tf.Variable(tf.random_uniform(shape=[self.input_size, self.hidden_size], minval=-1.0, maxval=1.0))
        b_1 = tf.Variable(tf.zeros([self.hidden_size]))
        inputs = tf.matmul(inputs, w_1) + b_1
        w_2 = tf.Variable(tf.random_uniform(shape=[self.hidden_size, 1], minval=-1.0, maxval=1.0))
        b_2 = tf.Variable(tf.zeros([1]))
        outputs = tf.matmul(inputs, w_2) + b_2
        outputs = tf.nn.relu(outputs)
        if self.keep_prob < 1.:
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
        return outputs


def _num_labels(weights):
    """Number of 1's in weights. Returns 1. if 0."""
    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)
    return num_labels


class CandidateSamplingLoss(object):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 num_candidate_samples=-1,
                 vocab_freqs=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_candidate_samples = num_candidate_samples
        self.vocab_freqs = vocab_freqs

    def __call__(self, x, labels, weights):
        nce_weights = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.vocab_size],
                                                      stddev=1.0 / math.sqrt(self.vocab_size)), name='lm_lin_w')
        nce_biases = tf.Variable(tf.zeros(shape=[self.vocab_size]), name='lm_lin_b')

        if self.num_candidate_samples > -1:
            assert self.vocab_freqs is not None
            labels = tf.expand_dims(labels, -1)
            sampled = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels,
                num_true=1,
                num_sampled=self.num_candidate_samples,
                unique=True,
                range_max=self.vocab_size,
                unigrams=self.vocab_freqs)

            lm_loss = tf.nn.sampled_softmax_loss(
                weights=tf.transpose(nce_weights),
                biases=nce_biases,
                labels=labels,
                inputs=x,
                num_sampled=self.num_candidate_samples,
                num_classes=self.vocab_size,
                sampled_values=sampled)
        else:
            logits = tf.matmul(x, nce_weights) + nce_biases
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

        lm_loss = tf.identity(
            tf.reduce_sum(lm_loss * weights) / _num_labels(weights),
            name='lm_xentropy_loss')
        return lm_loss


class ClassificationLoss(object):
    def __call__(self, logits, labels, weights):
        """Computes cross entropy loss between logits and labels.

        Args:
          logits: 2-D [timesteps*batch_size, m] float tensor, where m=1 if
            num_classes=2, otherwise m=num_classes.
          labels: 1-D [timesteps*batch_size] integer tensor.
          weights: 1-D [timesteps*batch_size] float tensor.

        Returns:
          Loss scalar of type float.
        """
        inner_dim = logits.get_shape().as_list()[-1]
        with tf.name_scope('classifier_loss'):
            # Logistic loss
            if inner_dim == 1:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.squeeze(logits), labels=tf.cast(labels, tf.float32))
            # Softmax loss
            else:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)

            num_lab = _num_labels(weights)
            cl_loss = tf.identity(tf.reduce_sum(weights * loss) / num_lab, name='classification_xentropy')
            return cl_loss