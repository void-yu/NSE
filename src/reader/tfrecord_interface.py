import random
import tensorflow as tf


class ShufflingTFRecordWriter(object):
    """Thin wrapper around TFRecordWriter that shuffles records."""

    def __init__(self, path):
        self._path = path
        self._records = []
        self._closed = False

    def write(self, record):
        assert not self._closed
        self._records.append(record)

    def close(self):
        assert not self._closed
        random.shuffle(self._records)
        with tf.python_io.TFRecordWriter(self._path) as f:
            for record in self._records:
                f.write(record)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        self.close()


# class PieceWrapper(object):
#     F_X = 'tokens'
#     F_Y = 'label'
#
#     def __init__(self):
#         self._seq = tf.train.Example()
#         self._flist = self._seq.feature_list

def make_example(tokens, label):
    tf.train.Example(features=tf.train.Features(feature={
        'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=[tokens])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }))
