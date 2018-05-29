import tensorflow as tf
import tensorlayer as tl
import time
import numpy as np
import pickle

# from src.naive_lstm.model_sent import LSTMModel
from src.naive_lstm.model_birnn import TextRNN

flags = tf.app.flags


flags.DEFINE_integer("save_every_n", 10, "")

flags.DEFINE_integer("embed_size", 300, "")
flags.DEFINE_integer("hidden_size", 512, "")
flags.DEFINE_integer("attn_lenth", 350, "")

flags.DEFINE_integer("vocab_size", 64189, "")
flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("seq_length", 100, "")
flags.DEFINE_integer("epoches", 25, "")

flags.DEFINE_float("learning_rate", 0.001, "")
flags.DEFINE_integer("decay_steps", 12000, "")
flags.DEFINE_float("decay_rate", 0.9, "")
flags.DEFINE_boolean("is_training", True, "")


flags.DEFINE_string("tensorb_dir", "D://Codes/NSE/src/naive_lstm/save/inited_refined_20_unigram_fixed/tensorb", "")
# flags.DEFINE_string("tensorb_dir", "D://Codes/NSE/src/naive_lstm/save/uninited_unigram", "")
flags.DEFINE_string("ckpt_dir", "D://Codes/NSE/src/naive_lstm/save/inited_refined_20_unigram_fixed/ckpt", "")
# flags.DEFINE_string("ckpt_dir", "D://Codes/NSE/src/naive_lstm/save/uninited_unigram", "")

FLAGS = flags.FLAGS


def restore_from_checkpoint(sess, saver, dir):
    ckpt = tf.train.get_checkpoint_state(dir)
    # print(ckpt)
    if not ckpt or not ckpt.model_checkpoint_path:
        print('No checkpoint found at {}'.format(dir))
        return False
    saver.restore(sess, ckpt.model_checkpoint_path)
    return True


def train_sent(sess):
    print("Read file --")
    start = time.time()

    # id2word, word2id = reader.read_glossary()
    # pretrained_wv = np.random.uniform(-1, 1, [FLAGS.vocab_size, FLAGS.embedding_size])
    PRETRAINED_VECS_PATH = 'D://Codes/NSE/data/used/embeddings/20-refined-word-picked.vec'
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
    w, pretrained_wv = get_vecs()


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


    # model = LSTMModel(
    #     seq_size=FLAGS.seq_lenth,
    #     vocab_size=FLAGS.vocab_size,
    #     embedding_size=FLAGS.embedding_size,
    #     hidden_size=FLAGS.hidden_size,
    #     attn_lenth=FLAGS.attn_lenth
    # )
    # model.buildTrainGraph()
    textRNN = TextRNN(1, 0.0001, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                      FLAGS.seq_length, FLAGS.vocab_size, FLAGS.embed_size, FLAGS.is_training, np.array(pretrained_wv))


    init = tf.global_variables_initializer()
    # sess.run(init, feed_dict={model.pretrained_wv: pretrained_wv})
    sess.run(init)

    saver_epoch = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
    saver_recent = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
    train_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/train', graph=sess.graph)
    valid_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/valid')
    # restore_from_checkpoint(sess, saver_recent, FLAGS.ckpt_dir)

    # global_step = 0

    curr_epoch = sess.run(textRNN.epoch_step)

    for epoch in range(curr_epoch, FLAGS.epoches):
        start_time = time.time()

        print('Epoch %d/%d' % (epoch + 1, FLAGS.epoches))
        for X_batch, y_batch in tl.iterate.minibatches(train_seqs, train_tars, batch_size=FLAGS.batch_size, shuffle=True):
            curr_loss, curr_acc, _, global_step, train_loss_scalar, train_acc_scalar = sess.run(
                [textRNN.loss_val, textRNN.accuracy, textRNN.train_op, textRNN.global_step, textRNN.loss_scalar, textRNN.acc_scalar],
                feed_dict={textRNN.input_x: X_batch,
                           textRNN.input_y: y_batch,
                           textRNN.dropout_keep_prob: 0.2})  # curr_acc--->TextCNN.accuracy -->,textRNN.dropout_keep_prob:1

            if global_step % 100 == 0:
                train_writer.add_summary(train_loss_scalar, global_step)
                train_writer.add_summary(train_acc_scalar, global_step)

                print("took %.5fs" % (time.time() - start_time))
                eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
                for X_valid_batch, y_valid_batch in tl.iterate.minibatches(valid_seqs, valid_tars, batch_size=FLAGS.batch_size, shuffle=True):
                    curr_eval_loss, logits, curr_eval_acc = sess.run(
                        [textRNN.loss_val, textRNN.logits, textRNN.accuracy],  # curr_eval_acc--->textCNN.accuracy
                        feed_dict={textRNN.input_x: X_valid_batch, textRNN.input_y: y_valid_batch, textRNN.dropout_keep_prob: 1})
                    eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
                valid_loss, valid_acc = eval_loss / float(eval_counter), eval_acc / float(eval_counter)
                print("Train loss: {:.4f}, train accuracy: {:.4f}".format(curr_loss, curr_acc))
                print("Test loss: {:.4f}, test accuracy: {:.4f}".format(valid_loss, valid_acc))

                saver_recent.save(sess, FLAGS.ckpt_dir + "/step{}.ckpt".format(global_step))
                valid_loss_scalar = tf.Summary(value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_loss)])
                valid_writer.add_summary(valid_loss_scalar, global_step)
                valid_acc_scalar = tf.Summary(value=[tf.Summary.Value(tag="valid_acc", simple_value=valid_acc)])
                valid_writer.add_summary(valid_acc_scalar, global_step)

        saver_epoch.save(sess, FLAGS.ckpt_dir + "/step{}.ckpt".format(epoch))

    end = time.time()
    print("Building model finished -- {:.4f} sec".format(end - start))



# def test_sent(sess):
#     print("Read file --")
#     start = time.time()
#
#     # id2word, word2id = reader.read_glossary()
#     # pretrained_wv = np.random.uniform(-1, 1, [FLAGS.vocab_size, FLAGS.embedding_size])
#
#     testfile = open('D://Codes/NSE/data/output/splited_test_data.pkl', 'rb')
#     testdata = pickle.load(testfile)
#     test_seqs = testdata['seqs']
#     test_tars = testdata['tars']
#
#     end = time.time()
#     print("Read finished -- {:.4f} sec".format(end-start))
#
#     # Build model
#     print("Building model --")
#
#     model = LSTMModel(
#         seq_size=FLAGS.seq_lenth,
#         vocab_size=FLAGS.vocab_size,
#         embedding_size=FLAGS.embedding_size,
#         hidden_size=FLAGS.hidden_size,
#         attn_lenth=FLAGS.attn_lenth
#     )
#     model.buildTrainGraph()
#
#     saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
#     # train_writer = tf.summary.FileWriter(logdir=FLAGS.tensorb_dir + '/train', graph=sess.graph)
#     restore_from_checkpoint(sess, saver, FLAGS.ckpt_dir)
#
#     i = 0
#     test_loss = list()
#     test_acc = list()
#     start_time = time.time()
#
#     for X_batch, y_batch in tl.iterate.minibatches(test_seqs, test_tars, batch_size=FLAGS.batch_size, shuffle=False):
#         i += 1
#
#         batch_loss, batch_acc = sess.run(
#             [model.loss, model.accuracy],
#             feed_dict={
#                 model.inputs: X_batch,
#                 model.labels: y_batch,
#             }
#         )
#         test_loss.append(batch_loss)
#         test_acc.append(batch_acc)
#
#     test_accuracy = np.mean(test_acc)
#     print('Test accuracy: %.5f' % test_accuracy)
#     print("took %.5fs" % (time.time() - start_time))



def main(_):
    with tf.Session() as sess:
        train_sent(sess)
        # test_sent(sess)

if __name__ == '__main__':
    tf.app.run()