import tensorflow as tf
import tensorlayer as tl
import time
import numpy as np
import pickle

from src.naive_lstm.model_sent import LSTMModel

flags = tf.app.flags


flags.DEFINE_integer("save_every_n", 10, "")

flags.DEFINE_integer("embedding_size", 300, "")
flags.DEFINE_integer("hidden_size", 500, "")
flags.DEFINE_integer("attn_lenth", 350, "")

flags.DEFINE_integer("vocab_size", 64189, "")
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_integer("seq_lenth", 50, "")
flags.DEFINE_integer("epoches", 25, "")

flags.DEFINE_string("tensorb_dir", "D://Codes/NSE/src/naive_lstm/save", "")
flags.DEFINE_string("ckpt_dir", "D://Codes/NSE/src/naive_lstm/save", "")

FLAGS = flags.FLAGS

def train_sent(sess):
    print("Read file --")
    start = time.time()

    # id2word, word2id = reader.read_glossary()
    pretrained_wv = np.random.uniform(-1, 1, [FLAGS.vocab_size, FLAGS.embedding_size])

    trainfile = open('D://Codes/NSE/data/output/splited_train_data.pkl', 'rb')
    traindata = pickle.load(trainfile)
    train_seqs = traindata['seqs']
    train_tars = traindata['tars']
    validfile = open('D://Codes/NSE/data/output/splited_valid_data.pkl', 'rb')
    validdata = pickle.load(validfile)
    valid_seqs = validdata['seqs']
    valid_tars = validdata['tars']

    end = time.time()
    print("Read finished -- {:.4f} sec".format(end-start))

    # Build model
    print("Building model --")
    start = end

    model = LSTMModel(
        seq_size=FLAGS.seq_lenth,
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        hidden_size=FLAGS.hidden_size,
        attn_lenth=FLAGS.attn_lenth
    )
    model.buildTrainGraph()

    init = tf.global_variables_initializer()
    sess.run(init, feed_dict={model.pretrained_wv: pretrained_wv})
    # sess.run(init)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
    # train_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/train', graph=sess.graph)


    i = 0
    for epoch in range(FLAGS.epoches):
        start_time = time.time()
        print('Epoch %d/%d' % (epoch + 1, FLAGS.epoches))
        for X_batch, y_batch in tl.iterate.minibatches(train_seqs, train_tars, batch_size=FLAGS.batch_size, shuffle=True):
            i += 1

            train_loss, _, train_scalar, train_acc = sess.run(
                [model.loss, model.train_op, model.train_scalar, model.accuracy],
                feed_dict={
                    model.inputs: X_batch,
                    model.labels: y_batch,
                }
            )

            if i % 100 == 0:
                print("took %.5fs" % (time.time() - start_time))
                valid_loss_list = []
                valid_acc_list = []
                for X_valid_batch, y_valid_batch in tl.iterate.minibatches(train_seqs, train_tars, batch_size=FLAGS.batch_size, shuffle=True):
                    valid_loss, valid_scalar, valid_acc = sess.run(
                        [model.loss, model.train_scalar, model.accuracy],
                        feed_dict={
                            model.inputs: X_valid_batch,
                            model.labels: y_valid_batch,
                        }
                    )
                    valid_loss_list.append(valid_loss)
                    valid_acc_list.append(valid_acc)
                print('Valid loss: %.5f' % np.mean(valid_loss_list), 'Valid accuracy: %.5f' % np.mean(valid_acc_list))
                saver.save(sess, FLAGS.ckpt_dir + "/step{}.ckpt".format(i))

        saver.save(sess, FLAGS.ckpt_dir + "/step{}.ckpt".format(epoch))


    end = time.time()
    print("Building model finished -- {:.4f} sec".format(end - start))

def main(_):
    with tf.Session() as sess:
        train_sent(sess)

if __name__ == '__main__':
    tf.app.run()