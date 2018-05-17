import pandas as pd
import numpy as np
from scipy import stats
import tensorlayer as tl

def get_vocab(wordpath):
    words = [word.strip() for word in open(wordpath).readlines()]
    word2index = {w: i for i, w in enumerate(words)}
    # vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/uninited/model_4.npz')[0]
    return words, word2index


def get_simlex999():
    vocab_filename = 'D://Codes/NSE/data/used/embeddings/word-picked'
    def make_vocab_ids(vocab_filename):
        with open(vocab_filename) as vocab_f:
            return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])
    path = 'D://Codes/NSE/data/raw/embeddingTest/simlex999.txt'
    with open(path) as f:
        evals = np.array(pd.read_csv(f, comment='#', sep='\t'))
        vocab = make_vocab_ids(vocab_filename)
        evals = [pair for pair in evals if pair[0] in vocab and pair[1] in vocab]
        print('Get %s pairs from simlex-999' % np.shape(evals)[0])
        return np.array(evals)


def get_wordsim353():
    vocab_filename = 'D://Codes/NSE/data/used/embeddings/word-picked'
    def make_vocab_ids(vocab_filename):
        with open(vocab_filename) as vocab_f:
            return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])
    path = 'D://Codes/NSE/data/raw/embeddingTest/wordsim353.tsv'
    with open(path) as f:
        evals = np.array(pd.read_csv(f, comment='#', sep='\t'))
        vocab = make_vocab_ids(vocab_filename)
        evals = [pair for pair in evals if pair[0] in vocab and pair[1] in vocab]
        print('Get %s pairs from wordsim-353' % np.shape(evals)[0])
        return np.array(evals)


def get_simverb3500():
    vocab_filename = 'D://Codes/NSE/data/used/embeddings/word-picked'
    def make_vocab_ids(vocab_filename):
        with open(vocab_filename) as vocab_f:
            return dict([(line.strip(), i) for i, line in enumerate(vocab_f)])
    path = 'D://Codes/NSE/data/raw/embeddingTest/simverb-3500-data/SimVerb-3500.txt'
    with open(path) as f:
        evals = f.readlines()
        evals = [i.split() for i in evals]
        evals = [[i[0], i[1], np.float32(i[3])] for i in evals]
        vocab = make_vocab_ids(vocab_filename)
        evals = [pair for pair in evals if pair[0] in vocab and pair[1] in vocab]
        print('Get %s pairs from SimVerb-3500' % np.shape(evals)[0])
        return evals



def eval_w2v_wordsim(words, word2index, vecs, simlex):
    results = []

    for pair in simlex:
        w1 = pair[0]
        w2 = pair[1]
        if w1 not in words or w2 not in words:
            print('Cant find', w1, w2)
            continue

        w1v = vecs[word2index[w1]]
        w2v = vecs[word2index[w2]]
        results.append([w1, w2, pair[2], np.sum(np.multiply(w1v, w2v))])

    print(stats.pearsonr([i[2] for i in results], [i[3] for i in results])[0], stats.spearmanr([i[2] for i in results], [i[3] for i in results]).correlation)


words, word2index = get_vocab('D://Codes/NSE/data/used/embeddings/word-picked')
turned_vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/inited_refined_2_unigram/model_0.npz')[0]
eval_w2v_wordsim(words, word2index, turned_vecs, get_wordsim353())