import csv
import os
import tensorflow as tf

from src.reader import tfrecord_interface

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

