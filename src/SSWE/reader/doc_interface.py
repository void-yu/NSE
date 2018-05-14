import os
import operator
import re
import random
from collections import namedtuple
import tensorflow as tf

# import data_utils

flags = tf.app.flags
FLAGS = flags.FLAGS


# IMDB
flags.DEFINE_string('imdb_input_dir', 'D://Codes/NSE/data/raw/aclImdb/', 'The input directory containing the '
                    'IMDB sentiment dataset.')
flags.DEFINE_integer('imdb_validation_pos_start_id', 10621, 'File id of the '
                     'first file in the pos sentiment validation set.')
flags.DEFINE_integer('imdb_validation_neg_start_id', 10625, 'File id of the '
                     'first file in the neg sentiment validation set.')


# The amazon reviews input file to use in either the RT or IMDB datasets.
flags.DEFINE_string('amazon_unlabeled_input_file', '',
                    'The unlabeled Amazon Reviews dataset input file. If set, '
                    'the input file is used to augment RT and IMDB vocab.')

Document = namedtuple('Document',
                      'content is_validation is_test label add_tokens')


EOS_TOKEN = '</s>'
PADDING_TOKEN = '</p>'

def documents(dataset='train',
              include_unlabeled=False,
              include_validation=False):

    if include_unlabeled and dataset != 'train':
        raise ValueError('If include_unlabeled=True, must use train dataset')

    random.seed(302)

    # ds = FLAGS.dataset
    # if ds == 'imdb':
    docs_gen = imdb_documents
    # else:
    #     raise ValueError('Unrecognized dataset %s' % FLAGS.dataset)
    for doc in docs_gen(dataset, include_unlabeled, include_validation):
        yield doc



def imdb_documents(dataset='train',
                   include_unlabeled=False,
                   include_validation=False):
    """Generates Documents for IMDB dataset.

    Data from http://ai.stanford.edu/~amaas/data/sentiment/

    Args:
        dataset: str, identifies folder within IMDB data directory, test or train.
        include_unlabeled: bool, whether to include the unsup directory. Only valid
            when dataset=train.
        include_validation: bool, whether to include validation data.

    Yields:
        Document

    Raises:
        ValueError: if FLAGS.imdb_input_dir is empty.
    """
    if not FLAGS.imdb_input_dir:
        raise ValueError('Must provide FLAGS.imdb_input_dir')

    tf.logging.info('Generating IMDB documents...')

    def check_is_validation(filename, class_label):
        if class_label is None:
            return False
        file_idx = int(filename.split('_')[0])
        is_pos_valid = (class_label and file_idx >= FLAGS.imdb_validation_pos_start_id)
        is_neg_valid = (not class_label and file_idx >= FLAGS.imdb_validation_neg_start_id)
        return is_pos_valid or is_neg_valid

    dirs = [(dataset + '/pos', True), (dataset + '/neg', False)]
    if include_unlabeled:
        dirs.append(('train/unsup', None))

    for d, class_label in dirs:
        for filename in os.listdir(os.path.join(FLAGS.imdb_input_dir, d)):
            is_validation = check_is_validation(filename, class_label)
            if is_validation and not include_validation:
                continue

            with open(os.path.join(FLAGS.imdb_input_dir, d, filename), encoding='utf8') as imdb_f:
                content = imdb_f.read()
            yield Document(
                content=content,
                is_validation=is_validation,
                is_test=False,
                label=class_label,
                add_tokens=True)

    if FLAGS.amazon_unlabeled_input_file and include_unlabeled:
        with open(FLAGS.amazon_unlabeled_input_file, encoding='utf8') as rt_f:
            for content in rt_f:
                yield Document(
                    content=content,
                    is_validation=False,
                    is_test=False,
                    label=None,
                    add_tokens=False)


def tokens(doc, ngram_join=True):
    if not (FLAGS.output_unigrams or FLAGS.output_bigrams or FLAGS.output_trigrams or FLAGS.output_char):
        raise ValueError(
            'At least one of {FLAGS.output_unigrams, FLAGS.output_bigrams, '
            'FLAGS.output_char} must be true')

    content = doc.content.strip()
    if FLAGS.lowercase:
        content = content.lower()

    if ngram_join:
        merge = lambda x: '_'.join(x)
    else:
        merge = lambda x: x

    # FLAGS.output_char
    if FLAGS.output_char:
        for char in content:
            yield char
    # FLAGS.output_unigrams or FLAGS.output_bigrams
    else:
        tokens_ = split_by_punct(content)
        for i, token in enumerate(tokens_):
            ngrams = {}
            if FLAGS.output_unigrams:
                ngrams['unigram'] = token
            if FLAGS.output_bigrams:
                previous_token = (tokens_[i-1] if i > 0 else EOS_TOKEN)
                bigram = merge([previous_token, token])
                ngrams['bigram'] = bigram
                if (i+1) == len(tokens_):
                    bigram = merge([token, EOS_TOKEN])
                    ngrams['bigram'] = bigram
            if FLAGS.output_trigrams:
                if i == 0:
                    continue
                elif i == 1:
                    previous_token = [EOS_TOKEN, tokens_[i-1]]
                else:
                    previous_token = [tokens_[i-2], tokens_[i-1]]
                previous_token.append(token)
                trigram = merge(previous_token)
                ngrams['trigram'] = trigram
                if (i+1) == len(tokens_):
                    trigram = merge([tokens_[i-1], token, EOS_TOKEN])
                    ngrams['trigram'] = trigram
            yield ngrams


def split_by_punct(segment):
    """Splits str segment by punctuation, filters our empties and spaces."""
    return [s for s in re.split(r'\W+', segment) if s and not s.isspace()]


def sort_vocab_by_frequency(vocab_freq_map):
    """Sorts vocab_freq_map by count.
    Args:
        vocab_freq_map: dict<str term, int count>, vocabulary terms with counts.

    Returns:
        list<tuple<str term, int count>> sorted by count, descending.
    """
    return sorted(
            vocab_freq_map.items(), key=operator.itemgetter(1), reverse=True)