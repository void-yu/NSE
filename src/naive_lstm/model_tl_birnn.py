import tensorlayer as tl
import tensorflow as tf
import numpy as np
import time
import pickle
import logging
import os

VOCAB_SIZE = 64191-2

EMBEDDING_SIZE = 300
HIDDEN_SIZE = 500

N_EPOCH = 5

BATCH_SIZE = 32

MODEL_FILE_PATH = 'save/tl_inited_refined_20_unigram/'
PRETRAINED_VECS_PATH = '../../data/used/embeddings/20-refined-word-picked.vec'



class MixEmbeddingInputlayer(tl.layers.Layer):

    def __init__(
            self,
            inputs,
            pretrained_vocab_size,
            uninited_vocabulary_size,
            embedding_size,
            pad_value=0,
            embeddings_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            embeddings_kwargs=None,
            name='average_embedding',
    ):

        super(MixEmbeddingInputlayer, self).__init__(prev_layer=None, name=name)
        logging.info("MixEmbeddingInputlayer %s: (%d, %d)" % (name, pretrained_vocab_size+uninited_vocabulary_size, embedding_size))

        # if embeddings_kwargs is None:
        #     embeddings_kwargs = {}

        if inputs.get_shape().ndims != 2:
            raise ValueError('inputs must be of size batch_size * batch_sentence_length')

        self.inputs = inputs

        with tf.variable_scope(name):
            self.pretrained_embeddings = tf.placeholder(tl.layers.LayersConfig.tf_dtype, shape=[pretrained_vocab_size, embedding_size], name='pretrained_embeddings')
            self.turnable_embeddings = tf.Variable(self.pretrained_embeddings, name='turnable_embeddings')
            self.uninited_embeddings = tf.get_variable(
                name='uninited_embeddings', shape=(uninited_vocabulary_size, embedding_size), initializer=embeddings_initializer,
                dtype=tl.layers.LayersConfig.tf_dtype,
                **(embeddings_kwargs or {})
            )

            mixed_embeddings = tf.concat([self.uninited_embeddings, self.turnable_embeddings], axis=0)

            word_embeddings = tf.nn.embedding_lookup(
                mixed_embeddings,
                self.inputs,
                name='word_embeddings',
            )

            masks = tf.not_equal(self.inputs, pad_value, name='masks')
            word_embeddings *= tf.cast(
                tf.expand_dims(masks, axis=-1),
                # tf.float32,
                dtype=tl.layers.LayersConfig.tf_dtype,
            )

        self.outputs = word_embeddings
        self.all_layers = [self.outputs]
        self.all_params = [self.turnable_embeddings, self.uninited_embeddings]
        self.all_drop = {}



class RNNMeanLayer(tl.layers.Layer):
    def __init__(
        self,
        layer=None,
        name='double_layer',
    ):
        super(RNNMeanLayer, self).__init__(prev_layer=layer, name=name)
        # 校验名字是否已被使用（不变）
        tl.layers.Layer.__init__(self, layer=layer, name=name)

        # 本层输入是上层的输出（不变）
        self.inputs = layer.outputs

        # 本层的功能实现（自定义部分）
        self.outputs = tf.reduce_mean(self.inputs, axis=1)

        # 更新层的参数（自定义部分）
        self.all_layers.append(self.outputs)





class BiLSTMClassifier(object):
    """Simple wrapper class for creating the graph of FastText classifier."""

    def __init__(self, vocab_size, embedding_size, hidden_size, is_train):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.is_train = is_train

        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        # Network structure
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            self.embedding_layer = MixEmbeddingInputlayer(
                inputs=self.inputs,
                pretrained_vocab_size=self.vocab_size,
                uninited_vocabulary_size=2,
                embedding_size=self.embedding_size)

            rnn_layer = tl.layers.DynamicRNNLayer(
                self.embedding_layer,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=self.hidden_size,
                dropout=(0.7 if self.is_train else None),
                sequence_length=tl.layers.retrieve_seq_length_op2(self.inputs),
                return_last=False,
                return_seq_2d=False)

            meaned_layer = RNNMeanLayer(rnn_layer)
            self.network = tl.layers.DenseLayer(meaned_layer, 2)

        # Training operation
        cost = tl.cost.cross_entropy(self.network.outputs, self.labels, name='cost')
        self.train_op = tf.train.AdamOptimizer().minimize(cost)

        # Predictions
        self.prediction_probs = tf.nn.softmax(self.network.outputs)
        self.predictions = tf.argmax(self.network.outputs, axis=1, output_type=tf.int32)

        # Evaluation
        are_predictions_correct = tf.equal(self.predictions, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(are_predictions_correct, tf.float32))

    def save(self, sess, filename):
        tl.files.save_npz(self.network.all_params, name=filename, sess=sess)

    def load(self, sess, filename):
        tl.files.load_and_assign_npz(sess, name=filename, network=self.network)



def get_vecs():
    vecs = []
    words = []
    with open(PRETRAINED_VECS_PATH, "r", encoding='utf8') as f:
        for line in f:
            s = line.split()
            if len(s) == 2:
                continue
            v = np.array([float(x) for x in s[1:]])
            if len(vecs) and vecs[-1].shape != v.shape:
                print("Got weird line", line)
                continue
            words.append(s[0])
            vecs.append(v)
    return words, vecs


def augment_with_ngrams(unigrams, unigram_vocab_size):
    """Augment unigram features with hashed n-gram features."""

    def get_ngrams(n):
        return list(zip(*[unigrams[i:] for i in range(n)]))

    return unigrams


def load_and_preprocess_imdb_data(n_gram=None):
    t_pkl = open('../../data/output/train_data_.pkl', 'rb')
    train = pickle.load(t_pkl)
    v_pkl = open('../../data/output/valid_data_.pkl', 'rb')
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
        X_train = np.array([augment_with_ngrams(x, VOCAB_SIZE) for x in X_train])
        X_valid = np.array([augment_with_ngrams(x, VOCAB_SIZE) for x in X_valid])
    return X_train, y_train, X_valid, y_valid


def load_and_preprocess_imdb_test_data(n_gram=None):
    t_pkl = open('../../data/output/test_data_.pkl', 'rb')
    test = pickle.load(t_pkl)
    X_test = []
    y_test = []
    for item in test:
        X_test.append(list(item['content']))
        y_test.append(1 if item['label'] is False else 0)
    if n_gram is not None:
        X_test = np.array([augment_with_ngrams(x, VOCAB_SIZE) for x in X_test])
    return X_test, y_test


def train_valid_and_save_model():
    X_train, y_train, X_valid, y_valid = load_and_preprocess_imdb_data()
    w, pretrained_wv = get_vecs()
    classifier = BiLSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        is_train=True
    )
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict={classifier.embedding_layer.pretrained_embeddings: pretrained_wv})
        # sess.run(init)

        i = 0
        for epoch in range(N_EPOCH):
            start_time = time.time()
            print('Epoch %d/%d' % (epoch + 1, N_EPOCH))
            for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True):
                i += 1

                sess.run(
                    classifier.train_op, feed_dict={
                        classifier.inputs: tl.prepro.pad_sequences(X_batch, value=0),
                        classifier.labels: y_batch,
                    }
                )

                if i % 100 == 0:
                    print("took %.5fs" % (time.time() - start_time))

                    valid_accuracy = sess.run(
                        classifier.accuracy, feed_dict={
                            classifier.inputs: tl.prepro.pad_sequences(X_valid, value=0),
                            classifier.labels: y_valid,
                        }
                    )
                    print('Valid accuracy: %.5f' % valid_accuracy)
                    classifier.save(sess, MODEL_FILE_PATH + 'model_' + str(i) + '.npz')
            classifier.save(sess, MODEL_FILE_PATH + 'model_' + str(epoch) + '.npz')
        classifier.save(sess, MODEL_FILE_PATH + 'model_' + str(i) + '.npz')


def load_and_test_model():
    X_test, y_test = load_and_preprocess_imdb_test_data()
    classifier = BiLSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        is_train=False
    )
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init, feed_dict={classifier.embeddings.pretrained_embeddings: pretrained_wv})
        # sess.run(init)
        foldername = MODEL_FILE_PATH
        pathDir = os.listdir(foldername)
        print(classifier.network)
        for filename in ['model_0.npz', 'model_1.npz', 'model_2.npz', 'model_3.npz', 'model_4.npz']:
            fullname = os.path.join('%s/%s' % (foldername, filename))
            print(fullname)
            classifier.load(sess, fullname)

            start_time = time.time()
            test_accuracy = []

            for X_batch, y_batch in tl.iterate.minibatches(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False):
                acc = sess.run(
                    classifier.accuracy, feed_dict={
                        classifier.inputs: tl.prepro.pad_sequences(X_batch, value=0),
                        classifier.labels: y_batch,
                    })
                test_accuracy.append(acc)

            test_accuracy = np.mean(test_accuracy)
            print('Test accuracy: %.5f' % test_accuracy)
            print("took %.5fs" % (time.time() - start_time))


if __name__ == '__main__':
    # train_valid_and_save_model()
    load_and_test_model()