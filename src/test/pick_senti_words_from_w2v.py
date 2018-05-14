import tensorflow as tf
import numpy as np

import we

def get_sentiwords():
    path = 'D://Codes/NSE/data/raw/SentiWords/SentiWords_1.1.txt'
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


def make_vocab_ids():
    vocab_filename = 'D://Codes/NSE/data/output/vocab_unigram.txt'
    with open(vocab_filename) as vocab_f:
        return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])


def run():
    new_words = []
    matrix = []
    coll = get_sentiwords()
    fasttext = we.WordEmbedding('D://Codes/NSE/data/raw/embeddings/fastText-crawl-300d-2M.vec')

    for i in coll:
        word = i[0]
        if word not in fasttext.words:
            # print(word, 'can not be found in fasttext')
            continue
        else:
            if word not in new_words:
                new_words.append(word)
                matrix.append(fasttext.vecs[fasttext.index[word]])

    f1 = open('D://Codes/NSE/data/used/embeddings/word-picked', 'w')
    f1.writelines("\n".join(new_words))
    f2 = open('D://Codes/NSE/data/used/embeddings/word-picked.vec', 'w')
    f2.write("\n".join([w + " " + " ".join([str(x) for x in v]) for w, v in zip(new_words, matrix)]))

run()