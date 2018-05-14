import os
from collections import defaultdict
import tensorflow as tf

import doc_interface

flags = tf.app.flags
FLAGS = flags.FLAGS

# Preprocessing config
flags.DEFINE_boolean('output_unigrams', False, 'Whether to output unigrams.')
flags.DEFINE_boolean('output_bigrams', False, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_trigrams', True, 'Whether to output trigrams.')
flags.DEFINE_boolean('output_char', False, 'Whether to output characters.')
flags.DEFINE_boolean('lowercase', True, 'Whether to lowercase document terms.')

flags.DEFINE_string('output_dir', 'D://Codes/NSE/data/output/', 'Path to the output folder.')
flags.DEFINE_string('vocab_file', 'D://Codes/NSE/data/output/vocab.txt', 'Path to the vocabulary file.')

flags.DEFINE_boolean('use_unlabeled', True, 'Whether to use the '
                     'unlabeled sentiment dataset in the vocabulary.')
flags.DEFINE_boolean('include_validation', True, 'Whether to include the '
                     'validation set in the vocabulary.')
flags.DEFINE_integer('doc_count_threshold', 1, 'The minimum number of '
                     'documents a word or bigram should occur in to keep '
                     'it in the vocabulary.')

MAX_VOCAB_SIZE = 100 * 1000


def fill_vocab_from_doc(doc, vocab_freqs, doc_counts):
    """Fills vocabulary and doc counts with tokens from doc.

    Args:
        doc: Document to read tokens from.
        vocab_freqs: dict<token, frequency count>
        doc_counts: dict<token, document count>

    Returns:
        None
    """
    doc_seen = set()

    for token in doc_interface.tokens(doc):
        if doc.add_tokens or token['trigram'] in vocab_freqs:
            vocab_freqs[token['trigram']] += 1
        if token['trigram'] not in doc_seen:
            doc_counts[token['trigram']] += 1
            doc_seen.add(token['trigram'])


def write_vocab_and_frequency(ordered_vocab_freqs, output_dir):
    """Writes ordered_vocab_freqs into vocab.txt and vocab_freq.txt."""
    tf.gfile.MakeDirs(output_dir)
    with open(os.path.join(output_dir, 'vocab_trigram.txt'), 'w', encoding='utf8') as vocab_f:
        with open(os.path.join(output_dir, 'vocab_trigram_freq.txt'), 'w', encoding='utf8') as freq_f:
            for word, freq in ordered_vocab_freqs:
                vocab_f.write('{}\n'.format(word))
                freq_f.write('{}\n'.format(freq))

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    vocab_freqs = defaultdict(int)
    doc_counts = defaultdict(int)

    # Fill vocabulary frequencies map and document counts map
    for doc in doc_interface.documents(
            dataset='train',
            include_unlabeled=FLAGS.use_unlabeled,
            include_validation=FLAGS.include_validation):
        fill_vocab_from_doc(doc, vocab_freqs, doc_counts)

    # Filter out low-occurring terms
    vocab_freqs = dict((term, freq) for term, freq in vocab_freqs.items()
                       if doc_counts[term] > FLAGS.doc_count_threshold)

    # Sort by frequency
    ordered_vocab_freqs = doc_interface.sort_vocab_by_frequency(vocab_freqs)

    # Limit vocab size
    ordered_vocab_freqs = ordered_vocab_freqs[:MAX_VOCAB_SIZE]

    # Add special tokens
    ordered_vocab_freqs.append((doc_interface.EOS_TOKEN, 1))
    ordered_vocab_freqs.append((doc_interface.PADDING_TOKEN, 1))

    # Write
    tf.gfile.MakeDirs(FLAGS.output_dir)
    write_vocab_and_frequency(ordered_vocab_freqs, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
