import numpy as np
import tensorflow as tf

import input_layer

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('TRAIN_CLASS', 'train_classification.tfrecords', '')
flags.DEFINE_string('TEST_CLASS', 'test_classification.tfrecords', '')
flags.DEFINE_string('VALID_CLASS', 'validate_classification.tfrecords', '')

flags.DEFINE_string('input_dir', 'D://Codes/NSE/data/output', 'Path to the output folder.')
flags.DEFINE_string('vocab_file', 'D://Codes/NSE/data/output/vocab_unigram.txt', 'Path to the vocabulary file.')
flags.DEFINE_string('vocab_freqs_file', 'D://Codes/NSE/data/output/vocab_unigram_freq.txt', 'Path to the vocabulary frequence file.')

flags.DEFINE_integer('num_classes', 2, 'Number of classes for classification')
flags.DEFINE_integer('batch_size', 128, 'Size of the batch.')
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
        self.fake_tokens = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, 1])
        fake_x = tf.split(tf.identity(x), num_or_size_splits=[1, 1, 1], axis=1)
        fake_x = tf.concat([fake_x[0], self.fake_tokens, fake_x[2]], axis=1)
        label = tf.cast(label, tf.float32)

        # lookup_x
        embedded = tf.nn.embedding_lookup(self.layers['embedding'], x)
        if self.embedding_keep_prob < 1.:
            embedded = tf.nn.dropout(embedded, keep_prob=self.embedding_keep_prob)
        embedded = tf.reshape(embedded, shape=[-1, self.embedding_size*self.window_size])

        # lookup_fake_x
        fake_embedded = tf.nn.embedding_lookup(self.layers['embedding'], fake_x)
        if self.embedding_keep_prob < 1.:
            fake_embedded = tf.nn.dropout(fake_embedded, keep_prob=self.embedding_keep_prob)
        fake_embedded = tf.reshape(fake_embedded, shape=[-1, self.embedding_size*self.window_size])

        # hTanh-linear_x
        lineared = tf.matmul(embedded, self.layers['W_1']) + self.layers['b_1']
        if self.linear_keep_prob < 1.:
            lineared = tf.nn.dropout(lineared, keep_prob=self.linear_keep_prob)
        nolineared = tf.nn.tanh(lineared)

        # hTanh-linear_fake_x
        fake_lineared = tf.matmul(fake_embedded, self.layers['W_1']) + self.layers['b_1']
        if self.linear_keep_prob < 1.:
            fake_lineared = tf.nn.dropout(fake_lineared, keep_prob=self.linear_keep_prob)
        fake_nolineared = tf.nn.tanh(fake_lineared)

        # last-linear_x
        expect = tf.matmul(nolineared, self.layers['W_2']) + self.layers['b_2']
        if self.linear_keep_prob < 1.:
            expect = tf.nn.dropout(expect, keep_prob=self.linear_keep_prob)

        # last-linear_fake_x
        fake_expect = tf.matmul(fake_nolineared, self.layers['W_2']) + self.layers['b_2']
        if self.linear_keep_prob < 1.:
            fake_expect = tf.nn.dropout(fake_expect, keep_prob=self.linear_keep_prob)

        # mixed-rank-loss
        syn_expect = expect[:, 0]
        sen_expect = expect[:, 1]
        fake_syn_expect = fake_expect[:, 0]
        fake_sen_expect = fake_expect[:, 1]
        syn_loss = tf.maximum(tf.zeros_like(syn_expect), tf.ones_like(syn_expect) - syn_expect + fake_syn_expect)
        sen_loss = tf.maximum(tf.zeros_like(sen_expect), tf.ones_like(sen_expect) - tf.multiply(label, sen_expect) + tf.multiply(label, fake_sen_expect))
        loss = self.loss_mix_weight * syn_loss + (1 - self.loss_mix_weight) * sen_loss
        # self.syn_expect = syn_expect
        # self.fake_syn_expect = fake_syn_expect
        # self.sen_expect = sen_expect
        # self.fake_sen_expect = fake_sen_expect
        self.loss = tf.reduce_mean(loss)

        self.opt = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)

        # collect_summary
        self.train_scalar = tf.summary.scalar('train_loss', loss)



def main(_):
    vocab_freqs = input_layer.get_vocab_freqs(
        path=FLAGS.vocab_freqs_file,
        vocab_size=FLAGS.vocab_size)
    model = SSWE_u(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        vocab_freqs=vocab_freqs,
        window_size=FLAGS.window_size,
        hidden_size=FLAGS.hidden_size,
        embedding_keep_prob=FLAGS.embedding_keep_prob,
        linear_keep_prob=FLAGS.linear_keep_prob,
        normalized=FLAGS.embedding_normalized,
        loss_mix_weight=FLAGS.loss_mix_weight)
    train_input = input_layer.get_batch(
        data_dir=FLAGS.input_dir,
        fname=FLAGS.TRAIN_CLASS,
        batch_size=FLAGS.batch_size)

    model(x=train_input['tokens'], label=train_input['label'])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                random_input = np.reshape(np.random.choice(FLAGS.vocab_size, FLAGS.batch_size), newshape=[-1, 1])

                # _, syn, fasyn, sen, fasen, loss = sess.run([model.opt, model.syn_expect, model.fake_syn_expect, model.sen_expect, model.fake_sen_expect, model.loss],
                #                                            feed_dict={model.fake_tokens: random_input})
                # print(syn, fasyn, sen, fasen, loss)
                _, loss = sess.run([model.opt, model.loss], feed_dict={model.fake_tokens: random_input})
                print(loss)

        except tf.errors.OutOfRangeError:
            tf.logging.info('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    tf.app.run()