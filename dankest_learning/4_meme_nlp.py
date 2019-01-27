import pickle
import re
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import os

def meme_nlp(meme_dir):
    p = Path(meme_dir)
    meme_files = list(p.glob('*.txt'))
    meme_files.sort(key=os.path.getmtime)
    meme_files = list(map(lambda x: x.stem, meme_files))[:2]
    splitter = re.compile(r"<sep>|\w+['`]\w+|\w+|\b[^\w\s]+|[^\w\s]", re.UNICODE | re.IGNORECASE)

    glove_embeddings = h5py.File('processed_data/glove_embeddings.hdf5')['glove_embeddings'][:]
    with open('processed_data/glove_vocabulary.p', 'rb') as f:
        glove_vocabulary = pickle.load(f)

    with open('processed_data/candidate_vocabulary.p', 'rb') as f:
        candidate_vocabulary = pickle.load(f)

    meme_vocabulary = OrderedDict()  # 0 is reserved for sequence masking
    meme_vocabulary['<end>'] = 1
    meme_vocabulary['<unk>'] = 2
    meme_vocabulary['<sep>'] = 3
    meme_vocabulary['<start>'] = 4
    glorot_init_val = np.sqrt(6/300)
    meme_embeddings = [
        np.zeros((300,), dtype=np.float32),  # masked...
        np.random.uniform(-glorot_init_val, glorot_init_val, size=(300,)).astype(np.float32),  # <end>, will be trained
        glove_embeddings[glove_vocabulary['<unk>']],  # <unk>
        np.random.uniform(-glorot_init_val, glorot_init_val, size=(300,)).astype(np.float32),  # <sep>, will be trained
        np.random.uniform(-glorot_init_val, glorot_init_val, size=(300,)).astype(np.float32)  # <start>, will be trained
    ]
    vocab_counter = 5

    for caption_file in tqdm(meme_files):
        for word in map(str.lower, caption_file.split('-')):
            if (word == '<sep>' or word in glove_vocabulary) \
                    and word in candidate_vocabulary and word not in meme_vocabulary:
                meme_vocabulary[word] = vocab_counter
                meme_embeddings.append(glove_embeddings[glove_vocabulary[word]])
                vocab_counter += 1
        with open(p.joinpath(f'{caption_file}.txt')) as f:
            for line in f:
                for word in map(str.lower, splitter.findall(line.strip())):
                    if (word == '<sep>' or word in glove_vocabulary) \
                            and word in candidate_vocabulary and word not in meme_vocabulary:
                        meme_vocabulary[word] = vocab_counter
                        meme_embeddings.append(glove_embeddings[glove_vocabulary[word]])
                        vocab_counter += 1

    return meme_vocabulary, meme_embeddings


print("4 - Generating actual embeddings/vocabulary of scraped dataset...")
meme_vocabulary, meme_embeddings = meme_nlp('memes')
print("Saving to processed_data/meme_vocabulary.p and processed_data/meme_embeddings.hdf5")
archive = h5py.File('processed_data/meme_embeddings.hdf5', 'w')
archive.create_dataset('meme_embeddings', data=meme_embeddings, compression="gzip", shuffle=True, fletcher32=True)

with open('processed_data/meme_vocabulary.p', 'wb') as f:
    pickle.dump(meme_vocabulary, f)
