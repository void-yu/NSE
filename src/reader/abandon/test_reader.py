import os
import tensorflow as tf

import data_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('ALL_LM', 'all_lm.tfrecords', '')
flags.DEFINE_string('TRAIN_LM', 'train_lm.tfrecords', '')
flags.DEFINE_string('TEST_LM', 'test_lm.tfrecords', '')
flags.DEFINE_string('ALL_CLASS', 'all_classification.tfrecords', '')
flags.DEFINE_string('TRAIN_CLASS', 'train_classification.tfrecords', '')
flags.DEFINE_string('TEST_CLASS', 'test_classification.tfrecords', '')
flags.DEFINE_string('VALID_CLASS', 'validate_classification.tfrecords', '')

flags.DEFINE_string('output_dir', 'D://Codes/NSE/data/output/', 'Path to the output folder.')
flags.DEFINE_string('vocab_file', 'D://Codes/NSE/data/output/vocab.txt', 'Path to the vocabulary file.')


def _read_single_sequence_example(file_list, tokens_shape=None):
    tf.logging.info('Constructing TFRecordReader from files: %s', file_list)
    file_queue = tf.train.string_input_producer(file_list)
    reader = tf.TFRecordReader()
    seq_key, serialized_record = reader.read(file_queue)
    context, sequence = tf.parse_single_sequence_example(
        serialized_record,
        sequence_features={
            data_utils.SequenceWrapper.F_TOKEN_ID:
                tf.FixedLenSequenceFeature(tokens_shape or [], dtype=tf.int64),
            data_utils.SequenceWrapper.F_LABEL:
                tf.FixedLenSequenceFeature([], dtype=tf.int64),
            data_utils.SequenceWrapper.F_WEIGHT:
                tf.FixedLenSequenceFeature([], dtype=tf.float32),})
    return seq_key, sequence


def main(_):
    data_path = os.path.join(FLAGS.output_dir, FLAGS.TRAIN_CLASS)
    if not tf.gfile.Exists(data_path):
        raise ValueError('Failed to find file: %s' % data_path)

    seq_key, sequence = _read_single_sequence_example([data_path], tokens_shape=[])

    label = sequence['label']
    token_id = sequence['token_id']
    weight = sequence['weight']
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for i in range(2):
            # while not coord.should_stop():
                print(sess.run([label, token_id, weight]))
        except tf.errors.OutOfRangeError:
            tf.logging.info('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


    def _get_tuple_state_names(num_states, base_name):
        """Returns state names for use with LSTM tuple state."""
        state_names = [('{}_{}_c'.format(i, base_name), '{}_{}_h'.format(i, base_name)) for i in range(num_states)]
        return state_names


if __name__ == '__main__':
    tf.app.run()