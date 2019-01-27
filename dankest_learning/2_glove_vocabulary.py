import pickle

import numpy as np


def get_vocab2idx(file_name):
    vocabs = np.genfromtxt(file_name, dtype=str, usecols=0, delimiter=' ', comments=None, autostrip=True)
    return {vocabs[i]: i for i in range(len(vocabs))}

print("2 - Generating GloVe vocabulary...")
glove_vocabulary = get_vocab2idx('GloVe/glove.42B.300d.txt')
print("Saving to processed_data/glove_vocabulary.p")
with open('processed_data/glove_vocabulary.p', 'wb') as f:
    pickle.dump(glove_vocabulary, f)

