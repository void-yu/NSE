import array
import hashlib
import time
import logging
import pickle
import os
import numpy as np
import tensorflow as tf

import tensorlayer as tl


# Hashed n-grams with 1 < n <= N_GRAM are included as features
# in addition to unigrams.
N_GRAM = 1

# Size of vocabulary; less frequent words will be treated as "unknown"
# VOCAB_SIZE = 100000
VOCAB_SIZE = 64191-2

# Number of buckets used for hashing n-grams
# N_BUCKETS = 100000+2
N_BUCKETS = 2

# Size of the embedding vectors
EMBEDDING_SIZE = 300

# Number of epochs for which the model is trained
N_EPOCH = 5

# Size of training mini-batches
BATCH_SIZE = 32

# Path to which to save the trained model
MODEL_FILE_PATH = 'save/demo/'
PRETRAINED_VECS_PATH = 'D://Codes/NSE/data/used/embeddings/10-refined-word-picked.vec'



class AverageMixEmbeddingInputlayer(tl.layers.Layer):

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

        super(AverageMixEmbeddingInputlayer, self).__init__(prev_layer=None, name=name)
        logging.info("MixEmbeddingInputlayer %s: (%d, %d)" % (name, pretrained_vocab_size+uninited_vocabulary_size, embedding_size))

        # if embeddings_kwargs is None:
        #     embeddings_kwargs = {}

        if inputs.get_shape().ndims != 2:
            raise ValueError('inputs must be of size batch_size * batch_sentence_length')

        self.inputs = inputs

        with tf.variable_scope(name):
            self.pretrained_embeddings = tf.placeholder(tl.layers.LayersConfig.tf_dtype, shape=[pretrained_vocab_size, embedding_size], name='pretrained_embeddings')
            self.uninited_embeddings = tf.get_variable(
                name='uninited_embeddings', shape=(uninited_vocabulary_size, embedding_size), initializer=embeddings_initializer,
                dtype=tl.layers.LayersConfig.tf_dtype,
                **(embeddings_kwargs or {})
                # **embeddings_kwargs
            )  # **(embeddings_kwargs or {}),

            mixed_embeddings = tf.concat([self.pretrained_embeddings, self.uninited_embeddings], axis=0)

            word_embeddings = tf.nn.embedding_lookup(
                mixed_embeddings,
                self.inputs,
                name='word_embeddings',
            )
            # Zero out embeddings of pad value
            masks = tf.not_equal(self.inputs, pad_value, name='masks')
            word_embeddings *= tf.cast(
                tf.expand_dims(masks, axis=-1),
                # tf.float32,
                dtype=tl.layers.LayersConfig.tf_dtype,
            )
            sum_word_embeddings = tf.reduce_sum(word_embeddings, axis=1)

            # Count number of non-padding words in each sentence
            sentence_lengths = tf.count_nonzero(
                masks,
                axis=1,
                keepdims=True,
                # dtype=tf.float32,
                dtype=tl.layers.LayersConfig.tf_dtype,
                name='sentence_lengths',
            )

            sentence_embeddings = tf.divide(
                sum_word_embeddings,
                sentence_lengths + 1e-8,  # Add epsilon to avoid dividing by 0
                name='sentence_embeddings'
            )

        self.outputs = sentence_embeddings
        self.all_layers = [self.outputs]
        self.all_params = [self.uninited_embeddings]
        self.all_drop = {}



class FastTextClassifier(object):
    """Simple wrapper class for creating the graph of FastText classifier."""

    def __init__(self, vocab_size, bucket_size, embedding_size, n_labels):
        self.vocab_size = vocab_size
        self.bucket_size = bucket_size
        self.embedding_size = embedding_size
        self.n_labels = n_labels

        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        # Network structure
        self.embeddings = AverageMixEmbeddingInputlayer(self.inputs, self.vocab_size, self.bucket_size, self.embedding_size)
        self.network = tl.layers.DenseLayer(self.embeddings, self.n_labels)
        # self.all_params = [self.embeddings.pretrained_embeddings] + self.network.all_params

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


def augment_with_ngrams(unigrams, unigram_vocab_size, n_buckets, n=2):
    """Augment unigram features with hashed n-gram features."""

    def get_ngrams(n):
        return list(zip(*[unigrams[i:] for i in range(n)]))

    def hash_ngram(ngram):
        bytes_ = array.array('L', ngram).tobytes()
        hash_ = int(hashlib.sha256(bytes_).hexdigest(), 16)
        return unigram_vocab_size + hash_ % n_buckets

    return unigrams + [hash_ngram(ngram) for i in range(2, n + 1) for ngram in get_ngrams(i)]


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
    w, pretrained_wv = get_vecs()
    classifier = FastTextClassifier(
        vocab_size=VOCAB_SIZE,
        bucket_size=N_BUCKETS,
        embedding_size=EMBEDDING_SIZE,
        n_labels=2
    )
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # sess.run(init)

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
                        classifier.embeddings.pretrained_embeddings: pretrained_wv
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
    # w, pretrained_wv = get_vecs()

    X_test, y_test = load_and_preprocess_imdb_test_data(N_GRAM)
    classifier = FastTextClassifier(
        vocab_size=VOCAB_SIZE,
        bucket_size=N_BUCKETS,
        embedding_size=EMBEDDING_SIZE,
        n_labels=2,
    )
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init, feed_dict={classifier.embeddings.pretrained_embeddings: pretrained_wv})
        # sess.run(init)
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


if __name__ == '__main__':
    with tf.device("/cpu:0"):
        train_valid_and_save_model()
        # load_and_test_model()