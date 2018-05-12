import string
import os
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('output_unigrams', True, 'Whether to output unigrams.')
flags.DEFINE_boolean('output_bigrams', True, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_trigrams', True, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_char', False, 'Whether to output characters.')
flags.DEFINE_boolean('lowercase', True, 'Whether to lowercase document terms.')
flags.DEFINE_string('output_dir', 'D://Codes/NSE/data/output/', 'Path to the output folder.')
flags.DEFINE_string('vocab_file', 'D://Codes/NSE/data/output/vocab.txt', 'Path to the vocabulary file.')


def make_vocab_ids(vocab_filename):
    if FLAGS.output_char:
        ret = dict([(char, i) for i, char in enumerate(string.printable)])
        ret['</s>'] = len(string.printable)
        return ret
    else:
        with open(vocab_filename) as vocab_f:
            return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])


vocab_ids = make_vocab_ids(FLAGS.vocab_file or os.path.join(FLAGS.output_dir, 'vocab.txt'))
# print(vocab_ids)
li = [i.strip() for i in open('D://Codes/NSE/data/raw/seeds-coling2014/positive-seeds.txt', 'r').readlines()]
print([word for word in li if word not in vocab_ids])