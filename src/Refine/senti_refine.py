from src.test import we
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy import stats



def save(words, vecs, filename):
    with open(filename, "w") as f:
        f.write("\n".join([w + " " + " ".join([str(x) for x in v]) for w, v in zip(words, vecs)]))
    print("Wrote to", filename)


def get_sentiwords():
    path = 'D://Codes/NSE/data/raw/SentiWords/SentiWords_1.1.txt'
    coll = []
    with open(path) as f:
        content = [line for line in f.readlines() if line[0] != '#']
        for line in content:
            pair = line.split()
            score = float(pair[1])
            pair = pair[0].split('#')
            pair.append(score)
            coll.append(pair)
    return coll

wordpair = [line.strip().split() for line in open('D://Codes/NSE/data/used/seeds/wordpairs-greater-than-0.5').readlines()]
sub_words_p = [line[0] for line in wordpair]
sub_words_n = [line[1] for line in wordpair]



def find_senti_direction(topK=1):
    picked = we.WordEmbedding('D://Codes/NSE/data/used/embeddings/fasttext/word-picked.vec')

    definitional = []

    for index in range(len(sub_words_p)):
        wp = sub_words_p[index]
        wn = sub_words_n[index]
        if wp not in picked.words or wn not in picked.words:
            print(wp, wn, 'not in word-picked.vec')
            continue
        definitional.append([wp, wn])

    pca = we.doPCA(definitional, picked)
    senti_direction = pca.components_[0]
    print(pca.explained_variance_ratio_)
    # dire_0 = picked.vecs[picked.index[definitional[4][0]]] - picked.vecs[picked.index[definitional[4][1]]]
    # dire_0 /= np.linalg.norm(dire_0)
    return senti_direction


def refine_word(lexicon, direction, alpha=0.5):
    picked = we.WordEmbedding('D://Codes/NSE/data/used/embeddings/fasttext/word-picked.vec')

    words = {}
    vecs = {}
    for word, _, score in lexicon:
        if word not in picked.words:
            continue
        elif word in words and abs(score) < abs(words[word]):
            continue
        else:
            words[word] = score
            vec = picked.vecs[picked.index[word]]
            vec_senti = np.sum(np.multiply(vec, direction)) / (np.linalg.norm(direction) * np.linalg.norm(direction)) * direction
            vec_vertical_to_senti = vec - vec_senti
            # print(np.linalg.norm(vec_senti))
            # print(np.linalg.norm(vec_vertical_to_senti))
            new_vec_senti = direction * score
            new_vec = vec_vertical_to_senti + alpha * new_vec_senti
            new_vec = new_vec / np.linalg.norm(new_vec)
            vecs[word] = new_vec
    # save(words.keys(), vecs.values(), 'D://Codes/NSE/data/used/embeddings/'+str(alpha)+'-refined-word-picked.vec')



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

lexicon = get_sentiwords()
dire = find_senti_direction(1)
# refine_word(lexicon, dire, alpha=20)


def eval_w2v_wordsim(simlex):
    refined = we.WordEmbedding('D://Codes/NSE/data/used/embeddings/10-refined-word-picked.vec')
    results = []

    for pair in simlex:
        w1 = pair[0]
        w2 = pair[1]
        if w1 not in refined.words or w2 not in refined.words:
            print('Cant find', w1, w2)
            continue

        w1v = refined.vecs[refined.index[w1]]
        w2v = refined.vecs[refined.index[w2]]
        results.append([w1, w2, pair[2], np.sum(np.multiply(w1v, w2v))])

    print(stats.pearsonr([i[2] for i in results], [i[3] for i in results])[0], stats.spearmanr([i[2] for i in results], [i[3] for i in results]).correlation)

    # for i in sorted(results, key=lambda x: x[3]):
    #     print(i)



# eval_w2v_wordsim(get_simverb3500())