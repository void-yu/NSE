import pandas as pd
import numpy as np
import csv
import os
import tensorflow as tf

from src.SSWE.reader import tfrecord_interface

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_batch(data_dir, fname, batch_size):
    data_path = os.path.join(data_dir, fname)
    if not tf.gfile.Exists(data_path):
        raise ValueError('Failed to find file: %s' % data_path)

    batch = tfrecord_interface._read_and_batch(
        data_path=data_path,
        batch_size=batch_size
    )
    return batch


def get_vocab_freqs(path, vocab_size):
    if tf.gfile.Exists(path):
        with tf.gfile.Open(path) as f:
            # Get pre-calculated frequencies of words.
            reader = csv.reader(f, quoting=csv.QUOTE_NONE)
            freqs = [int(row[-1]) for row in reader]
            if len(freqs) != vocab_size:
                raise ValueError('Frequency file length %d != vocab size %d' %
                                 (len(freqs), vocab_size))
    else:
        freqs = [1] * vocab_size
    return freqs


def get_simlex999(vocab_filename):
    path = FLAGS.simlex_999
    if not tf.gfile.Exists(path):
        raise ValueError('Failed to find file: %s' % path)
    else:
        with tf.gfile.Open(path) as f:
            evals = np.array(pd.read_csv(f, comment='#', sep='\t'))
            vocab = make_vocab_ids(vocab_filename)
            evals = [[vocab[pair[0]], vocab[pair[1]], pair[2]] for pair in evals if pair[0] in vocab and pair[1] in vocab]
            print('Get %s pairs from simlex-999' % np.shape(evals)[0])
            return np.array(evals)


def make_vocab_ids(vocab_filename):
    with open(vocab_filename) as vocab_f:
        return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])

flags.DEFINE_string('senti_word', 'D://Codes/NSE/data/raw/SentiWords/SentiWords_1.1.txt', '')


def get_sentiwords():
    path = FLAGS.senti_word
    if not tf.gfile.Exists(path):
        raise ValueError('Failed to find file: %s' % path)
    else:
        coll = []
        with tf.gfile.Open(path) as f:
            content = [line for line in f.readlines() if line[0] != '#']
            for line in content:
                pair = line.split()
                score = float(pair[1])
                pair = pair[0].split('#')
                pair.append(score)
                coll.append(pair)
        return coll

# pairs = get_sentiwords()
# print(len(pairs))

