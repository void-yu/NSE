import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

# Classifier
flags.DEFINE_integer('num_classes', 2, 'Number of classes for classification')
flags.DEFINE_integer('batch_size', 64, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 100, 'Number of timesteps for BPTT')

flags.DEFINE_integer('vocab_size', 71457, '')
flags.DEFINE_integer('embedding_size', 50, '')
flags.DEFINE_integer('window_size', 3, '')
flags.DEFINE_integer('hidden_size', 20, '')
flags.DEFINE_float('embedding_keep_prob', 1.0, '')
flags.DEFINE_float('linear_keep_prob', 1.0, '')
flags.DEFINE_float('loss_mix_weight', 0.5, '')


class SSWE_u(object):
    def _normalize(self, emb, vocab_size, vocab_freqs):
        self.vocab_freqs = tf.constant(vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keepdims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keepdims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev



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

        w_1 = tf.Variable(tf.random_uniform(shape=[self.embedding_size * self.window_size, self.hidden_size], minval=-1.0, maxval=1.0))
        b_1 = tf.Variable(tf.zeros([self.hidden_size]))
        self.layers['w1'] = w_1
        self.layers['b1'] = b_1


    # each x is a trigram whose element is also a n-gram
    # shape(x) = [batch_size, 3]
    def __call__(self, x, fake_x, label):
        # lookup_x
        embedded = tf.nn.embedding_lookup(self.layers['embedding'], x)
        if self.embedding_keep_prob < 1.:
            embedded = tf.nn.dropout(embedded, keep_prob=self.embedding_keep_prob)
        embedded = tf.concat(embedded, axis=-1)

        # lookup_fake_x
        fake_embedded = tf.nn.embedding_lookup(self.layers['embedding'], fake_x)
        if self.embedding_keep_prob < 1.:
            fake_embedded = tf.nn.dropout(fake_embedded, keep_prob=self.embedding_keep_prob)
        fake_embedded = tf.concat(fake_embedded, axis=-1)

        # hTanh-linear_x
        lineared = tf.matmul(embedded, self.layers['w_1']) + self.layers['b_1']
        if self.linear_keep_prob < 1.:
            lineared = tf.nn.dropout(lineared, keep_prob=self.linear_keep_prob)
        expect = tf.nn.tanh(lineared)

        # hTanh-linear
        fake_lineared = tf.matmul(fake_embedded, self.layers['w_1']) + self.layers['b_1']
        if self.linear_keep_prob < 1.:
            fake_lineared = tf.nn.dropout(fake_lineared, keep_prob=self.linear_keep_prob)
        fake_expect = tf.nn.tanh(fake_lineared)

        # syntactic-rank-loss
        loss_syntactic = tf.maximum(0., 1. - expect + fake_expect)
        loss_sentiment = tf.maximum(0., 1. - label*expect + label*fake_expect)
        loss = self.loss_mix_weight * loss_syntactic + (1 - self.loss_mix_weight) * loss_sentiment
        loss = tf.reduce_mean(loss)

        opt = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)

        # collect_summary
        self.train_scalar = tf.summary.scalar('train_loss', loss)


SSWE_u(vocab_size=FLAGS.vocab_size,
       embedding_size=FLAGS.embedding_size,
       vocab_freqs=FLAGS.vocab_freqs,
       window_size=FLAGS.window_size,
       hidden_size=FLAGS.hidden_size,
       embedding_keep_prob=FLAGS.embedding_keep_prob,
       linear_keep_prob=FLAGS.linear_keep_prob,
       normalized=FLAGS.normalized,
       loss_mix_weight=FLAGS.loss_mix_weight)