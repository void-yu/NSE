import string
import os
import tensorflow as tf

import doc_interface
import data_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# Preprocessing config
flags.DEFINE_boolean('output_unigrams', True, 'Whether to output unigrams.')
flags.DEFINE_boolean('output_bigrams', True, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_trigrams', True, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_char', False, 'Whether to output characters.')
flags.DEFINE_boolean('lowercase', True, 'Whether to lowercase document terms.')

flags.DEFINE_string('ALL_LM', 'all_lm.tfrecords', '')
flags.DEFINE_string('TRAIN_LM', 'train_lm.tfrecords', '')
flags.DEFINE_string('TEST_LM', 'test_lm.tfrecords', '')
flags.DEFINE_string('ALL_CLASS', 'all_classification.tfrecords', '')
flags.DEFINE_string('TRAIN_CLASS', 'train_classification.tfrecords', '')
flags.DEFINE_string('TEST_CLASS', 'test_classification.tfrecords', '')
flags.DEFINE_string('VALID_CLASS', 'validate_classification.tfrecords', '')

flags.DEFINE_string('output_dir', 'D://Codes/NSE/data/output/', 'Path to the output folder.')
flags.DEFINE_string('vocab_file', 'D://Codes/NSE/data/output/vocab.txt', 'Path to the vocabulary file.')


def build_shuffling_tf_record_writer(fname):
    return data_utils.ShufflingTFRecordWriter(os.path.join(FLAGS.output_dir, fname))


def build_tf_record_writer(fname):
    return tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, fname))


def build_input_sequence(doc, vocab_ids):
    """Builds input sequence from file.

    Splits lines on whitespace. Treats punctuation as whitespace. For word-level
    sequences, only keeps terms that are in the vocab.

    Terms are added as token in the SequenceExample. The EOS_TOKEN is also
    appended. Label and weight features are set to 0.

    Args:
        doc: Document (defined in `document_generators`) from which to build the
            sequence.
        vocab_ids: dict<term, id>.

    Returns:
        SequenceExampleWrapper.
    """
    seqs = []
    if FLAGS.output_unigrams:
        seq_unigram = data_utils.SequenceWrapper()
    if FLAGS.output_bigrams:
        seq_bigram = data_utils.SequenceWrapper()
    if FLAGS.output_trigrams:
        seq_trigram = data_utils.SequenceWrapper()

    for token in doc_interface.tokens(doc, ngram_join=True):
        if FLAGS.output_unigrams and token['unigram'] in vocab_ids:
            seq_unigram.add_timestep().set_token(vocab_ids[token['unigram']])
        if FLAGS.output_bigrams and token['bigram'] in vocab_ids:
            seq_bigram.add_timestep().set_token(vocab_ids[token['bigram']])
        if FLAGS.output_trigrams and token['trigram'] in vocab_ids:
            seq_trigram.add_timestep().set_token(vocab_ids[token['trigram']])

    # Add EOS token to end
    if FLAGS.output_unigrams:
        seq_unigram.add_timestep().set_token(vocab_ids[data_utils.EOS_TOKEN])
        seqs.append(seq_unigram)
    if FLAGS.output_bigrams:
        seq_bigram.add_timestep().set_token(vocab_ids[data_utils.EOS_TOKEN])
        seqs.append(seq_bigram)
    if FLAGS.output_trigrams:
        seq_trigram.add_timestep().set_token(vocab_ids[data_utils.EOS_TOKEN])
        seqs.append(seq_trigram)

    return seqs


def make_vocab_ids(vocab_filename):
    if FLAGS.output_char:
        ret = dict([(char, i) for i, char in enumerate(string.printable)])
        ret[data_utils.EOS_TOKEN] = len(string.printable)
        return ret
    else:
        with open(vocab_filename) as vocab_f:
            return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])


def generate_lm_training_data(vocab_ids, writer_lm_all):
    writer_lm = build_shuffling_tf_record_writer(FLAGS.ALL_CLASS)

    for doc in doc_interface.documents(
            dataset='train', include_unlabeled=True, include_validation=True):
        input_seq = build_input_sequence(doc, vocab_ids)
        if len(input_seq) < data_utils.MIN_LENGTH:
            continue
        lm_seq = data_utils.build_lm_sequence(input_seq)

        lm_seq_ser = lm_seq.seq.SerializeToString()
        writer_lm_all.write(lm_seq_ser)
        if not doc.is_validation:
            writer_lm.write(lm_seq_ser)

    writer_lm.close()


def generate_lm_test_data(vocab_ids, writer_lm_all):
    writer_lm = build_shuffling_tf_record_writer(FLAGS.TEST_LM)

    for doc in doc_interface.documents(
            dataset='test', include_unlabeled=False, include_validation=True):
        input_seq = build_input_sequence(doc, vocab_ids)
        if len(input_seq) < data_utils.MIN_LENGTH:
            continue
        lm_seq = data_utils.build_lm_sequence(input_seq)

        lm_seq_ser = lm_seq.seq.SerializeToString()
        writer_lm_all.write(lm_seq_ser)

    writer_lm.close()


def generate_labeled_seq(vocab_ids, writer_class_all):
    writer_class_train = build_shuffling_tf_record_writer(FLAGS.TRAIN_CLASS)
    writer_class_valid = build_shuffling_tf_record_writer(FLAGS.VALID_CLASS)
    writer_class_test = build_shuffling_tf_record_writer(FLAGS.TEST_CLASS)

    for doc in doc_interface.documents(dataset='train', include_unlabeled=False, include_validation=True):
        seqs = build_input_sequence(doc, vocab_ids)
        for input_seq in seqs:
            if len(input_seq) < data_utils.MIN_LENGTH:
                continue

            if doc.label is not None:
                label_seq = data_utils.build_labeled_sequence(
                    input_seq,
                    doc.label,
                    weight_type='Unrestricted')
                writer_class_all.write(label_seq.seq.SerializeToString())
                if not doc.is_validation and not doc.is_test:
                    writer_class_train.write(label_seq.seq.SerializeToString())
                elif doc.is_validation:
                    writer_class_valid.write(label_seq.seq.SerializeToString())
                elif doc.is_test:
                    writer_class_test.write(label_seq.seq.SerializeToString())

    writer_class_all.close()
    writer_class_train.close()
    writer_class_valid.close()
    writer_class_test.close()


def test_seq(vocab_ids):
    for doc in doc_interface.documents(dataset='train', include_unlabeled=False, include_validation=True):
        input_seq = build_input_sequence(doc, vocab_ids)
        for seq in input_seq:
            print(seq. labels)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Assigning vocabulary ids...')

    vocab_ids = make_vocab_ids(FLAGS.vocab_file or os.path.join(FLAGS.output_dir, 'vocab.txt'))

    writer_cl_all = build_shuffling_tf_record_writer(FLAGS.ALL_CLASS)

    generate_labeled_seq(vocab_ids, writer_cl_all)

    # test_seq(vocab_ids)


    # tf.logging.info('Generating training data...')
    # generate_lm_training_data(vocab_ids, writer_lm_all)

    # tf.logging.info('Generating test data...')
    # generate_lm_test_data(vocab_ids, writer_lm_all)

if __name__ == '__main__':
    tf.app.run()