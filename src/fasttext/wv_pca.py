from src.test import we
from sklearn.decomposition import PCA
import tensorlayer as tl
import numpy as np


def get_vecs(wordpath, vecpath):
    words = [word.strip() for word in open(wordpath).readlines()]
    word2index = {w: i for i, w in enumerate(words)}
    # vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/uninited/model_4.npz')[0]
    vecs = we.WordEmbedding(vecpath).vecs
    print(np.shape(vecs))
    return words, word2index, vecs


def analysis_pca(words, word2index, vecs):
    wordpair = [line.strip().split() for line in open('D://Codes/NSE/data/used/seeds/wordpairs-greater-than-0.5').readlines()]
    sub_words_p = [line[0] for line in wordpair]
    sub_words_n = [line[1] for line in wordpair]

    matrix = []
    for index in range(len(sub_words_p)):
        w1 = sub_words_p[index]
        w2 = sub_words_n[index]
        if w1 not in words or w2 not in words:
            print(w1, w2)
            continue
        wv_p = vecs[word2index[w1]]
        wv_n = vecs[word2index[w2]]
        center = (wv_p + wv_n) / 2
        matrix.append(wv_p - center)
        matrix.append(wv_n - center)
    pca = PCA(n_components=10)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_)


def find_senti_direction(words, word2index, vecs, topK=1):
    wordpair = [line.strip().split() for line in open('D://Codes/NSE/data/used/seeds/wordpairs-greater-than-0.5').readlines()]
    sub_words_p = [line[0] for line in wordpair]
    sub_words_n = [line[1] for line in wordpair]

    definitional = []
    matrix = []

    for index in range(len(sub_words_p)):
        wp = sub_words_p[index]
        wn = sub_words_n[index]
        if wp not in words or wn not in words:
            print(wp, wn, 'not in word-picked.vec')
            continue
        definitional.append([wp, wn])
        wv_p = vecs[word2index[wp]]
        wv_n = vecs[word2index[wn]]
        center = (wv_p + wv_n) / 2
        matrix.append(wv_p - center)
        matrix.append(wv_n - center)

    pca = PCA(n_components=10)
    pca.fit(matrix)
    senti_direction = pca.components_[0]
    print(pca.explained_variance_ratio_)
    # dire_0 = picked.vecs[picked.index[definitional[4][0]]] - picked.vecs[picked.index[definitional[4][1]]]
    # dire_0 /= np.linalg.norm(dire_0)
    return senti_direction


words, word2index, vecs = get_vecs(wordpath='D://Codes/NSE/data/used/embeddings/word-picked',
                                   vecpath='D://Codes/NSE/data/used/embeddings/10-refined-word-picked.vec')
turned_vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/demo/model_300.npz')[0]
turned_vecs_2 = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/inited_refined_2_unigram/model_400.npz')[0]

sd_ori = find_senti_direction(words, word2index, vecs)
sd_turned = find_senti_direction(words, word2index, turned_vecs)
sd_turned_2 = find_senti_direction(words, word2index, turned_vecs_2)
print(np.linalg.norm(sd_ori), np.linalg.norm(sd_turned))
print(np.sum(np.multiply(sd_turned_2, sd_turned)))

# analysis_pca(words, word2index, turned_vecs)