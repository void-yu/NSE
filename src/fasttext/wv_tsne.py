import numpy as np
from sklearn.manifold import TSNE
import time
import tensorlayer as tl
import matplotlib.pyplot as plt

vocab = [word.strip() for word in open('D://Codes/NSE/data/used/embeddings/word-picked').readlines()]
word2index = {w: i for i, w in enumerate(vocab)}
turned_vecs = tl.files.load_npz(name='D://Codes/NSE/src/fasttext/save/inited_refined_20_unigram/model_4.npz')[0]

# fig = plt.figure(figsize=(8, 8))
#
# t0 = time.time()
# tsne = TSNE(n_components=2, init='pca', random_state=0)
# Y = tsne.fit_transform(turned_vecs)
# print('t-SNE: %.2g sec' % (time.time() - t0))
# ax = fig.add_subplot(2, 1, 2)
# plt.scatter(Y[:, 0], Y[:, 1], c=color)