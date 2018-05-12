import random
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

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


def make_example(tokens, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))



def _read_single_example(file_list):
    tf.logging.info('Constructing TFRecordReader from files: %s', file_list)
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    key, serialized_record = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_record,
        features={
            'tokens': tf.FixedLenFeature([FLAGS.window_size], dtype=tf.int64),
            'label': tf.FixedLenFeature([], dtype=tf.int64)
        })
    return key, features


def _read_and_batch(data_path, batch_size):
    key, features = _read_single_example([data_path])

    batch = tf.train.batch(
        tensors=features,
        batch_size=batch_size,
    )
    return batch