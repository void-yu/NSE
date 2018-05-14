import we
import numpy as np

vecs = []
words = []

with open('D://Codes/NSE/data/used/embeddings/word-picked.vec', "r", encoding='utf8') as f:
    for line in f:
        s = line.split()
        if len(s) == 2:
            continue
        v = np.array([float(x) for x in s[1:]])
        if len(vecs) and vecs[-1].shape != v.shape:
            print("Got weird line", line)
            continue
        #                 v /= np.linalg.norm(v)
        words.append(s[0])
        vecs.append(v)

f.close()
words_dict = dict([(line.strip(), i) for i, line in enumerate(words)])
words_set = set(words)

with open('D://Codes/NSE/data/used/embeddings/word-picked.vec', "w", encoding='utf8') as f:
    for word in words_set:
        vec = [str(item) for item in vecs[words_dict[word]]]
        f.write(word + ' ' + ' '.join(vec) + '\n')
