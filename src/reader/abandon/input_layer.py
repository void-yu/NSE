import os
import tensorflow as tf

from src.reader.abandon import data_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# Classifier
flags.DEFINE_integer('num_classes', 2, 'Number of classes for classification')
flags.DEFINE_integer('window_size', 3, 'Number of classes for classification')
flags.DEFINE_integer('batch_size', 64, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 100, 'Number of timesteps for BPTT')

flags.DEFINE_string('output_dir', 'D://Codes/NSE/data/output/', 'Path to the output folder.')

flags.DEFINE_string('TRAIN_CLASS', 'train_classification.tfrecords', '')



def _read_single_sequence_example(file_list, tokens_shape=None):
    """Reads and parses SequenceExamples from TFRecord-encoded file_list."""
    tf.logging.info('Constructing TFRecordReader from files: %s', file_list)
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    seq_key, serialized_record = reader.read(file_queue)
    ctx, sequence = tf.parse_single_sequence_example(
        serialized_record,
        sequence_features={
        data_utils.SequenceWrapper.F_TOKEN_ID:
            tf.FixedLenSequenceFeature(tokens_shape or [], dtype=tf.int64),
        data_utils.SequenceWrapper.F_LABEL:
            tf.FixedLenSequenceFeature([], dtype=tf.int64),
        data_utils.SequenceWrapper.F_WEIGHT:
            tf.FixedLenSequenceFeature([], dtype=tf.float32),
        })
    return seq_key, ctx, sequence



def _generate_batch(data_dir,
                    fname,
                    batch_size):
    data_path = os.path.join(data_dir, fname)
    if not tf.gfile.Exists(data_path):
        raise ValueError('Failed to find file: %s' % data_path)

    tokens_shape = []
    seq_key, ctx, sequence = _read_single_sequence_example([data_path], tokens_shape=tokens_shape)

    batch = tf.train.batch(sequence, batch_size)
    return batch


sequence = _generate_batch(data_dir=FLAGS.output_dir,
                    fname=FLAGS.TRAIN_CLASS,
                    batch_size=FLAGS.batch_size)
label = sequence['label']
token_id = sequence['token_id']
weight = sequence['weight']


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for i in range(10):
            # while not coord.should_stop():
            print(sess.run([label, token_id, weight]))
    except tf.errors.OutOfRangeError:
        tf.logging.info('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()