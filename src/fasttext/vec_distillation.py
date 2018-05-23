from src.test import we
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy import stats
import pickle


def find_naive_senti_direction(words, word2index, vecs):
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


def find_P_N_vecs(vocab, word2index, vecs, lexicon):
    wp = {}
    vecs_p = {}
    wn = {}
    vecs_n = {}
    for word, _, score in lexicon:
        if word not in vocab:
            continue
        elif word in wp and abs(score) < abs(wp[word]):
            continue
        elif word in wn and abs(score) < abs(wn[word]):
            continue
        else:
            if score < 0:
                wn[word] = score
                vecs_n[word] = vecs[word2index[word]]
            elif score > 0:
                wp[word] = score
                vecs_p[word] = vecs[word2index[word]]

    pickle.dump({'wn': wn, 'vecn': vecs_n, 'wp': wp, 'vecp': vecs_p}, open('D://Codes/NSE/data/used/embeddings/p_n_vecs_inited_refined_20_unigram_4.pkl', 'wb'))


def find_p_n_PCA_senti_directions(topk=1):
    p_n_vecs = pickle.load(open('D://Codes/NSE/data/used/embeddings/p_n_vecs_inited_refined_20_unigram_4.pkl', 'rb'))
    wn = p_n_vecs['wn']
    wp = p_n_vecs['wp']
    vecn = p_n_vecs['vecn']
    vecp = p_n_vecs['vecp']
    matrix = list(vecn.values()) + list(vecp.values())
    print(np.shape(matrix))
    pca = PCA(n_components=10)
    pca.fit(matrix)
    senti_direction_list = pca.components_[:topk]
    print(pca.explained_variance_ratio_)
    # dire_0 = picked.vecs[picked.index[definitional[4][0]]] - picked.vecs[picked.index[definitional[4][1]]]
    # dire_0 /= np.linalg.norm(dire_0)
    return senti_direction_list


def find_random_PCA_senti_directions(vecs, topk=1):
    indexs = np.random.choice(64189, 40)
    matrix = vecs[indexs]
    print(np.shape(matrix))
    pca = PCA(n_components=10)
    pca.fit(matrix)
    random_senti_direction_list = pca.components_[:topk]
    print(pca.explained_variance_ratio_)
    # dire_0 = picked.vecs[picked.index[definitional[4][0]]] - picked.vecs[picked.index[definitional[4][1]]]
    # dire_0 /= np.linalg.norm(dire_0)
    return random_senti_direction_list


def find_p_n_mean_senti_direction():
    p_n_vecs = pickle.load(open('D://Codes/NSE/data/used/embeddings/p_n_vecs_inited_refined_20_unigram_4.pkl', 'rb'))
    wn = p_n_vecs['wn']
    wp = p_n_vecs['wp']
    vecn = p_n_vecs['vecn'].values()
    vecp = p_n_vecs['vecp'].values()
    vp = np.zeros([300])
    vn = np.zeros([300])
    for vec in vecp:
        vp += vec
    for vec in vecn:
        vn += vec
    vp /= len(vecp)
    vn /= len(vecn)
    # dire_0 = picked.vecs[picked.index[definitional[4][0]]] - picked.vecs[picked.index[definitional[4][1]]]
    # dire_0 /= np.linalg.norm(dire_0)
    return (vp-vn)/np.linalg.norm(vp-vn)


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



def p_n_lda():
    p_n_vecs = pickle.load(open('D://Codes/NSE/data/used/embeddings/p_n_vecs_inited_refined_20_unigram_4.pkl', 'rb'))
    wn = p_n_vecs['wn']
    wp = p_n_vecs['wp']
    c1 = []
    c2 = []
    for vec in p_n_vecs['vecn'].values():
        c1.append(vec)
    for vec in p_n_vecs['vecp'].values():
        c2.append(vec)
    c1 = np.array(c1)
    c2 = np.array(c2)
    # -*- coding: UTF-8 -*-
    # c1 第一类样本，每行是一个样本
    # c2 第二类样本，每行是一个样本

    # 计算各类样本的均值和所有样本均值
    m1 = np.mean(c1, axis=0)  # 第一类样本均值
    m2 = np.mean(c2, axis=0)  # 第二类样本均值
    c = np.vstack((c1, c2))  # 所有样本
    m = np.mean(c, axis=0)  # 所有样本的均值

    # 计算类内离散度矩阵Sw
    n1 = c1.shape[0]  # 第一类样本数
    n2 = c2.shape[0]  # 第二类样本数
    # 求第一类样本的散列矩阵s1
    s1 = 0
    for i in range(0, n1):
        s1 = s1 + (c1[i, :] - m1).T * (c1[i, :] - m1)
    # 求第二类样本的散列矩阵s2
    s2 = 0
    for i in range(0, n2):
        s2 = s2 + (c2[i, :] - m2).T * (c2[i, :] - m2)
    Sw = (n1 * s1 + n2 * s2) / (n1 + n2)
    # 计算类间离散度矩阵Sb
    Sb = (n1 * (m - m1).T * (m - m1) + n2 * (m - m2).T * (m - m2)) / (n1 + n2)
    # 求最大特征值对应的特征向量
    eigvalue, eigvector = np.linalg.eig(np.mat(Sw).I * Sb)  # 特征值和特征向量
    indexVec = np.argsort(-eigvalue)  # 对eigvalue从大到小排序，返回索引
    nLargestIndex = indexVec[:1]  # 取出最大的特征值的索引
    explained_variance_ratio_ = eigvalue[nLargestIndex] / np.sum(eigvalue)
    print(explained_variance_ratio_)
    senti_dire = eigvector[:, nLargestIndex]  # 取出最大的特征值对应的特征向量
    return senti_dire.getA().flatten()



def eval_p_n_principle_dire_similarity():
    p_n_vecs = pickle.load(open('D://Codes/NSE/data/used/embeddings/p_n_vecs_inited_refined_20_unigram_4.pkl', 'rb'))
    vecn = [list(i) for i in p_n_vecs['vecn'].values()]
    print(np.shape(vecn))
    pca_n = PCA(n_components=10)
    pca_n.fit(vecn)
    sendire_n = pca_n.components_[0]
    print(pca_n.explained_variance_ratio_)

    vecp = [list(i) for i in p_n_vecs['vecp'].values()]
    print(np.shape(vecp))
    pca_p = PCA(n_components=10)
    pca_p.fit(vecp)
    sendire_p = pca_p.components_[0]
    print(pca_p.explained_variance_ratio_)

    print('Similarity between positive direction and negative direction', np.sum(np.multiply(sendire_n, sendire_p)))
    return sendire_n, sendire_p


def distill(vecs, direction, savepath, beta=0):
    new_vecs = []
    for vec in vecs:
        vec_senti = np.sum(np.multiply(vec, direction)) / (
                    np.linalg.norm(direction) * np.linalg.norm(direction)) * direction
        vec_vertical_to_senti = vec - vec_senti
        # print(np.shape(vec_senti))
        new_vec = beta * vec_vertical_to_senti + vec_senti
        # print(np.shape(new_vec))
        new_vec = new_vec / np.linalg.norm(new_vec)
        new_vecs.append(new_vec)
    np.save(savepath, new_vecs)


def distill_multi(vecs, directions, savepath):
    new_vecs = []
    for vec in vecs:
        new_vec = np.zeros_like(vec)
        for dire in directions:
            vec_senti = np.sum(np.multiply(vec, dire)) / (
                        np.linalg.norm(dire) * np.linalg.norm(dire)) * dire
            new_vec += vec_senti
        new_vec = new_vec / np.linalg.norm(new_vec)
        new_vecs.append(new_vec)
    np.save(savepath, new_vecs)


def rev_distill(vecs, direction, savepath, gama=0.05):
    new_vecs = []
    for vec in vecs:
        vec_senti = np.sum(np.multiply(vec, direction)) / (
                    np.linalg.norm(direction) * np.linalg.norm(direction)) * direction
        vec_vertical_to_senti = vec - vec_senti
        # print(np.shape(vec_senti))
        new_vec = vec_vertical_to_senti + gama * vec_senti
        # print(np.shape(new_vec))
        new_vec = new_vec / np.linalg.norm(new_vec)
        new_vecs.append(new_vec)
    np.save(savepath, new_vecs)


def ver_length_norm(vecs, savepath):
    new_vecs = []
    for vec in vecs:
        new_vecs.append(vec / np.linalg.norm(vec))
    np.save(savepath, new_vecs)



import tensorflow as tf
import tensorlayer as tl


vocab = [word.strip() for word in open('D://Codes/NSE/data/used/embeddings/word-picked').readlines()]
word2index = {w: i for i, w in enumerate(vocab)}
turned_vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/inited_refined_20_unigram/model_4.npz')[0]
#
# lexicon = get_sentiwords()
# find_P_N_vecs(vocab, word2index, turned_vecs, lexicon)

# dire_naive = find_naive_senti_direction(vocab, word2index, turned_vecs) # score: 0.74241
# dire_mean = find_p_n_mean_senti_direction() # score: 0.53045
dire_PCA_list = find_p_n_PCA_senti_directions(topk=150) # score: 0.74010
# dire_randomPCA_list = find_random_PCA_senti_directions(turned_vecs, topk=5) # score: 0.74
# dire_lda = p_n_lda()

dire_n, dire_p = eval_p_n_principle_dire_similarity() # score: 0.78620

# distill(turned_vecs, dire_PCA_list[0], 'D://Codes/NSE/src/fasttext/save/inited_refined_20_unigram/model_4_distilled', beta=0)
distill_multi(turned_vecs, dire_PCA_list, 'D://Codes/NSE/src/fasttext/save/inited_refined_20_unigram/model_4_distilled')
# rev_distill(turned_vecs, dire_mean, 'D://Codes/NSE/src/fasttext/save/inited_refined_20_unigram/model_4_rev_distilled', gama=0)
# ver_length_norm(turned_vecs, 'D://Codes/NSE/src/fasttext/save/inited_refined_20_unigram/model_4_normed')


# print(np.sum(np.multiply(dire_randomPCA_, dire_randomPCA)))


