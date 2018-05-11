import tensorflow as tf
import os

import doc_interface
import tfrecord_interface

flags = tf.app.flags
FLAGS = flags.FLAGS

# Preprocessing config
flags.DEFINE_boolean('output_unigrams', True, 'Whether to output unigrams.')
flags.DEFINE_boolean('output_bigrams', False, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_trigrams', False, 'Whether to output bigrams.')
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
flags.DEFINE_string('vocab_file', 'D://Codes/NSE/data/output/vocab_unigram.txt', 'Path to the vocabulary file.')


def make_vocab_ids(vocab_filename):
    with open(vocab_filename) as vocab_f:
        return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])

vocab_ids = make_vocab_ids(FLAGS.vocab_file or os.path.join(FLAGS.output_dir, 'vocab.txt'))

def generate_unigram():
    train_data = []
    valid_data = []
    test_data = []
    for doc in doc_interface.documents(dataset='train', include_unlabeled=True, include_validation=True):
        if doc.label is None:
            continue

        if doc.is_validation:
            coll = valid_data
        elif doc.is_test:
            coll = test_data
        else:
            coll = train_data

        sequence = []
        for token in doc_interface.tokens(doc, ngram_join=True):
            token_ = token['unigram']
            if FLAGS.output_unigrams and token_ in vocab_ids:
                sequence.append(vocab_ids[token_])

        for i, token in enumerate(sequence):
            previous_token = (sequence[i - 1] if i > 0 else doc_interface.PADDING_TOKEN)
            post_token = (sequence[i + 1] if i < (len(sequence)+1) else doc_interface.EOS_TOKEN)
            tokens = [previous_token, token, post_token]

        ex = make_example(image, label)
        writer.write(ex.SerializeToString())

generate_unigram()

