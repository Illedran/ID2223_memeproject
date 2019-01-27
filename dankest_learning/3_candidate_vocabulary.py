import pickle
import re
from pathlib import Path

from tqdm import tqdm
import os

def generate_candidate(meme_dir):
    p = Path(meme_dir)
    meme_files = list(p.glob('*.txt'))
    meme_files.sort(key=os.path.getmtime)
    meme_files = list(map(lambda x: x.stem, meme_files))[:2]
    splitter = re.compile(r"<sep>|\w+['`]\w+|\w+|\b[^\w\s]+|[^\w\s]", re.UNICODE | re.IGNORECASE)

    # First pass: generate dictionary of candidate words
    removed = set()
    candidate_vocabulary = set()
    for caption_file in tqdm(meme_files):
        # print(caption_file)
        with open(p.joinpath(f'{caption_file}.txt')) as f:
            label_words = caption_file.split('-')
            for word in label_words:
                if word not in removed:
                    if word not in candidate_vocabulary:
                        removed.add(word)
                else:
                    candidate_vocabulary.add(word)
            for line in f:
                words = map(str.lower, splitter.findall(line.strip()))
                for word in words:
                    if word not in removed:
                        if word not in candidate_vocabulary:
                            removed.add(word)
                    else:
                        candidate_vocabulary.add(word)

    candidate_vocabulary.add('<start>')
    candidate_vocabulary.add('<sep>')
    candidate_vocabulary.add('<end>')
    candidate_vocabulary.add('<unk>')
    return candidate_vocabulary, removed


print("3 - Generating candidate vocabulary from dataset...")
candidate_vocabulary, removed = generate_candidate('memes')
print(f"Removed {len(removed)} single words.")
print("Saving to processed_data/candidate_vocabulary.p")
with open('processed_data/candidate_vocabulary.p', 'wb') as f:
    pickle.dump(candidate_vocabulary, f)
