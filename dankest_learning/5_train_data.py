import pickle
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import os

def train_data(meme_dir):
    p = Path(meme_dir)
    meme_files = list(p.glob('*.txt'))
    meme_files.sort(key=os.path.getmtime)
    meme_files = list(map(lambda x: x.stem, meme_files))[:2]
    splitter = re.compile(r"<sep>|\w+['`]\w+|\w+|\b[^\w\s]+|[^\w\s]", re.UNICODE | re.IGNORECASE)
    total_train_data = 0

    with open('processed_data/meme_vocabulary.p', 'rb') as f:
        meme_vocabulary = pickle.load(f)

    img_label_embeddings = {}
    caption_embeddings_per_meme = {}

    for caption_file in tqdm(meme_files):
        # Generate label embedding
        label_idx = []
        for word in map(str.lower, caption_file.split('-')):
            if word in meme_vocabulary:
                label_idx.append(meme_vocabulary[word])
            else:
                label_idx.append(meme_vocabulary['<unk>'])
        label_embedding = pad_sequences([label_idx], 25, value=0, padding='post', truncating='post')

        # Generate caption embedding
        captions = []
        with open(p.joinpath(f'{caption_file}.txt')) as f:
            for line in f:
                indexed_caption = [meme_vocabulary['<start>']]
                unk_counter = 0
                for word in map(str.lower, splitter.findall(line.strip())):
                    # print(word)
                    if word in meme_vocabulary:
                        indexed_caption.append(meme_vocabulary[word])
                    else:
                        unk_counter += 1
                        indexed_caption.append(meme_vocabulary['<unk>'])
                        if unk_counter >= 2:
                            break
                if unk_counter >= 2:
                    continue
                else:
                    indexed_caption.append(meme_vocabulary['<end>'])
                    captions.append(indexed_caption)
            caption_embeddings_per_meme[caption_file] = captions
            total_train_data += len(captions)

        # Generate image embedding
        img_embedding = preprocess_input(
            np.expand_dims(cv2.resize(np.array(load_img(p.joinpath(f'{caption_file}.png'))), (216, 216)), axis=0)) \
            .astype(np.float32)

        img_label_embeddings[caption_file] = (img_embedding, label_embedding)

    vocab_size = len(meme_vocabulary) + 1
    print(total_train_data)
    X = h5py.File('processed_data/X.hdf5', 'w')
    X_img = X.create_dataset('X_img',
                             shape=(total_train_data, 216, 216, 3),
                             maxshape=(total_train_data, 216, 216, 3),
                             dtype=np.float32,
                             compression="lzf",
                             shuffle=True,
                             # fletcher32=True
                             )
    X_label = X.create_dataset('X_label',
                               shape=(total_train_data, 25),
                               maxshape=(total_train_data, 25),
                               dtype=np.int32,
                               compression="lzf",
                               shuffle=True,
                               # fletcher32=True
                               )
    X_caption = X.create_dataset('X_caption',
                                 shape=(total_train_data, 25),
                                 maxshape=(total_train_data, 25),
                                 dtype=np.int32,
                                 compression="lzf",
                                 shuffle=True,
                                 # fletcher32=True
                                 )

    y = h5py.File('processed_data/y.hdf5', 'w').create_dataset('y',
                                                               shape=(total_train_data, 25, vocab_size),
                                                               maxshape=(total_train_data, 25, vocab_size),
                                                               dtype=np.bool,
                                                               compression="lzf",
                                                               shuffle=True,
                                                               # fletcher32=True
                                                               )

    print("Writing data to HDF5...")
    i = 0
    for l in tqdm(meme_files):
        padded = pad_sequences(caption_embeddings_per_meme[l], 25, value=0, padding='post', truncating='post')
        X_img[i:i + len(padded)] = np.tile(img_label_embeddings[l][0], (len(padded), 1, 1, 1))
        X_label[i:i + len(padded)] = np.tile(img_label_embeddings[l][1], (len(padded), 1))
        X_caption[i:i + len(padded)] = padded
        y_tmp = np.zeros((len(padded), 25, vocab_size), dtype=np.bool)
        for j, m in enumerate(padded):
            for k, word_id in enumerate(m[1:]):  # Shift 1 i.e. predict next word
                y_tmp[j][k][word_id] = True
        y[i:i+len(padded)] = y_tmp
        i += len(padded)

    with open('processed_data/inference_meme_embeddings.p', 'wb') as f:
        pickle.dump(img_label_embeddings, f)


print("5 - Generating training data...")
train_data('memes')
