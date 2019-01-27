import h5py
import numpy as np
from tqdm import tqdm


def iterfromtxt(file_name, col_split, max_col, delimiter=' ', dtype=np.float32, **kwargs):
    """
    Like np.genfromtxt, but better.
    """

    def iter_func():
        with open(file_name) as f:
            for l in tqdm(f, total=1917495):
                s = l.strip().split(delimiter)
                for item in s[col_split:max_col]:
                    yield dtype(item)

    data = np.fromiter(iter_func(), dtype=dtype, **kwargs)
    data = data.reshape((-1, max_col - col_split))
    return data

print("1 - Parsing GloVe embeddings...")
glove_embeddings = iterfromtxt('GloVe/glove.42B.300d.txt', col_split=1, max_col=301)
print("Saving to processed_data/glove_embeddings.hdf5")
archive = h5py.File('processed_data/glove_embeddings.hdf5', 'w')
archive.create_dataset('glove_embeddings', data=glove_embeddings, compression="gzip", shuffle=True, fletcher32=True)
