import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import matplotlib.pyplot as plt


from tensorflow.contrib.tensorboard.plugins import projector

def restore_from_checkpoint(sess, saver, dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    # print(ckpt)
    if not ckpt or not ckpt.model_checkpoint_path:
        print('No checkpoint found at {}'.format(dir))
        return False
    saver.restore(sess, ckpt.model_checkpoint_path)
    return True


# Create randomly initialized embedding weights which will be trained.
N = 10000 # Number of items (vocab size).
D = 200 # Dimensionality of the embedding.
LOG_DIR = 'log'

import we
# vecs = we.WordEmbedding('D://Codes/NSE/data/used/embeddings/10-refined-word-picked.vec').vecs

# turned_vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/uninited_unigram/model_4.npz')[0][:-2]
# turned_vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/inited_refined_2_unigram/model_4.npz')[0]
# print(np.shape(turned_vecs))
# embedding_var = tf.Variable(turned_vecs, name='word_embedding')

with tf.name_scope('embeddings'), tf.variable_scope('embeddings'):
    embedding_var = tf.Variable(tf.truncated_normal([64189, 300], stddev=0.1), name='embeddings')

with tf.Session() as sess:
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    restore_from_checkpoint(sess, saver, 'D://Codes/NSE/src/naive_lstm/save/inited_unigram')

    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'word-picked.tsv'

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)


# from tensorflow.examples.tutorials.mnist import input_data
#
# LOG_DIR = 'logs'
#
# mnist = input_data.read_data_sets('data/MNIST_data')
# images = tf.Variable(mnist.test.images, name='images')
#
# with tf.Session() as sess:
#     saver = tf.train.Saver([images])
#
#     sess.run(images.initializer)
#     saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))