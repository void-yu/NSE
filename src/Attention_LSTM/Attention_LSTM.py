import numpy as np
import tensorflow as tf
from scipy import stats

import input_layer

K = tf.contrib.keras

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



class Attention_LSTM(object):
    def __init__(self, cl_logits_input_dim=None):
        self.global_step = tf.train.get_or_create_global_step()
        self.vocab_freqs = _get_vocab_freqs()

        # Cache VatxtInput objects
        self.cl_inputs = None
        self.lm_inputs = None

        # Cache intermediate Tensors that are reused
        self.tensors = {}

        # Construct layers which are reused in constructing the LM and
        # Classification graphs. Instantiating them all once here ensures that
        # variable reuse works correctly.
        self.layers = {}
        self.layers['embedding'] = Embedding(
            FLAGS.vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
            self.vocab_freqs, FLAGS.keep_prob_emb)
        self.layers['lstm'] = LSTM(
            FLAGS.rnn_cell_size, FLAGS.rnn_num_layers, FLAGS.keep_prob_lstm_out)
        self.layers['lm_loss'] = SoftmaxLoss(
            FLAGS.vocab_size,
            FLAGS.num_candidate_samples,
            self.vocab_freqs,
            name='LM_loss')

        cl_logits_input_dim = cl_logits_input_dim or FLAGS.rnn_cell_size
        self.layers['cl_logits'] = cl_logits_subgraph(
            [FLAGS.cl_hidden_size] * FLAGS.cl_num_layers, cl_logits_input_dim,
            FLAGS.num_classes, FLAGS.keep_prob_cl_hidden)


    def classifier_graph(self, inputs):
        """Constructs classifier graph from inputs to classifier loss.

        * Caches the VatxtInput object in `self.cl_inputs`
        * Caches tensors: `cl_embedded`, `cl_logits`, `cl_loss`

        Returns:
            loss: scalar float.
        """
        # inputs = _inputs('train', pretrain=False)
        self.cl_inputs = inputs
        embedded = self.layers['embedding'](inputs.tokens)
        self.tensors['cl_embedded'] = embedded

        _, next_state, logits, loss = self.cl_loss_from_embedding(
                embedded, return_intermediates=True)
        tf.summary.scalar('classification_loss', loss)
        self.tensors['cl_logits'] = logits
        self.tensors['cl_loss'] = loss

        acc = accuracy(logits, inputs.labels, inputs.weights)
        tf.summary.scalar('accuracy', acc)

        adv_loss = (self.adversarial_loss() * tf.constant(
                FLAGS.adv_reg_coeff, name='adv_reg_coeff'))
        tf.summary.scalar('adversarial_loss', adv_loss)

        total_loss = loss + adv_loss
        tf.summary.scalar('total_classification_loss', total_loss)

        with tf.control_dependencies([inputs.save_state(next_state)]):
            total_loss = tf.identity(total_loss)

        return total_loss


class Embedding(K.layers.Layer):
    """Embedding layer with frequency-based normalization and dropout."""

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 normalize=False,
                 vocab_freqs=None,
                 keep_prob=1.,
                 **kwargs):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.normalized = normalize
        self.keep_prob = keep_prob

        if normalize:
            assert vocab_freqs is not None
            self.vocab_freqs = tf.constant(
                vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))

        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.device('/cpu:0'):
            self.var = self.add_weight(
                shape=(self.vocab_size, self.embedding_dim),
                initializer=tf.random_uniform_initializer(-1., 1.),
                name='embedding')

        if self.normalized:
            self.var = self._normalize(self.var)

        super(Embedding, self).build(input_shape)

    def call(self, x):
        embedded = tf.nn.embedding_lookup(self.var, x)
        if self.keep_prob < 1.:
            shape = embedded.get_shape().as_list()

            embedded = tf.nn.dropout(
                embedded, self.keep_prob, noise_shape=(shape[0], 1, shape[2]))
        return embedded

    def _normalize(self, emb):
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev
    
    
class LSTM(object):
    """LSTM layer using static_rnn.

    Exposes variables in `trainable_weights` property.
    """

    def __init__(self, cell_size, num_layers=1, keep_prob=1., name='LSTM'):
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, initial_state, seq_length):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(
                        self.cell_size,
                        forget_bias=0.0,
                        reuse=tf.get_variable_scope().reuse)
                for _ in range(self.num_layers)
            ])

            # shape(x) = (batch_size, num_timesteps, embedding_dim)
            # Convert into a time-major list for static_rnn
            x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

            lstm_out, next_state = tf.contrib.rnn.static_rnn(
                    cell, x, initial_state=initial_state, sequence_length=seq_length)

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


class SoftmaxLoss(K.layers.Layer):
    """Softmax xentropy loss with candidate sampling."""

    def __init__(self,
                 vocab_size,
                 num_candidate_samples=-1,
                 vocab_freqs=None,
                 **kwargs):
        self.vocab_size = vocab_size
        self.num_candidate_samples = num_candidate_samples
        self.vocab_freqs = vocab_freqs
        super(SoftmaxLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[0]
        with tf.device('/cpu:0'):
            self.lin_w = self.add_weight(
                shape=(input_shape[-1], self.vocab_size),
                name='lm_lin_w',
                initializer=K.initializers.glorot_uniform())
            self.lin_b = self.add_weight(
                shape=(self.vocab_size,),
                name='lm_lin_b',
                initializer=K.initializers.glorot_uniform())

        super(SoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        x, labels, weights = inputs
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
                    weights=tf.transpose(self.lin_w),
                    biases=self.lin_b,
                    labels=labels,
                    inputs=x,
                    num_sampled=self.num_candidate_samples,
                    num_classes=self.vocab_size,
                    sampled_values=sampled)
        else:
            logits = tf.matmul(x, self.lin_w) + self.lin_b
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)

        lm_loss = tf.identity(
                tf.reduce_sum(lm_loss * weights) / _num_labels(weights),
                name='lm_xentropy_loss')
        return lm_loss
    
    
def _num_labels(weights):
    """Number of 1's in weights. Returns 1. if 0."""
    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)
    return num_labels


def cl_logits_subgraph(layer_sizes, input_size, num_classes, keep_prob=1.):
    """Construct multiple ReLU layers with dropout and a linear layer."""
    subgraph = K.models.Sequential(name='cl_logits')
    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            subgraph.add(
                K.layers.Dense(layer_size, activation='relu', input_dim=input_size))
        else:
            subgraph.add(K.layers.Dense(layer_size, activation='relu'))

        if keep_prob < 1.:
            subgraph.add(K.layers.Dropout(1. - keep_prob))
    subgraph.add(K.layers.Dense(1 if num_classes == 2 else num_classes))
    return subgraph


def cl_loss_from_embedding(self,
                           embedded,
                           inputs=None,
                           return_intermediates=False):
      """Compute classification loss from embedding.

      Args:
          embedded: 3-D float Tensor [batch_size, num_timesteps, embedding_dim]
          inputs: VatxtInput, defaults to self.cl_inputs.
          return_intermediates: bool, whether to return intermediate tensors or only
              the final loss.

      Returns:
          If return_intermediates is True:
              lstm_out, next_state, logits, loss
          Else:
              loss
      """
      if inputs is None:
          inputs = self.cl_inputs

      lstm_out, next_state = self.layers['lstm'](embedded, inputs.state, inputs.length)
      logits = self.layers['cl_logits'](lstm_out)
      loss = classification_loss(logits, inputs.labels, inputs.weights)

      if return_intermediates:
          return lstm_out, next_state, logits, loss
      else:
          return loss
      


def classification_loss(logits, labels, weights):
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
        tf.summary.scalar('num_labels', num_lab)
        return tf.identity(
                tf.reduce_sum(weights * loss) / num_lab, name='classification_xentropy')