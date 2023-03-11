import json
import random
from collections import Counter

import spacy
import torch
from torchtext import datasets
from tqdm import tqdm

MAP_LABELS = {"neg": 1, "pos": 2}

SEED = 1234
MAX_VOCAB_SIZE = 25000


def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_data, test_data = datasets.IMDB(root="./imdb/", split=('train', 'test'))

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])

    train = []
    cnt = 0
    for item in tqdm(train_data):
        label = MAP_LABELS[item[0]]
        text = item[1]
        doc = nlp(text)
        words = [w.text for w in doc]
        train.append({"id": cnt, "text": text, "label": label, "words": words})
        cnt += 1

    id_range = list(range(len(train)))
    train_range = set(random.sample(id_range, int(len(id_range) * 0.7)))

    new_train = []
    val = []
    for item in train:
        if item["id"] in train_range:
            new_train.append(item)
        else:
            val.append(item)

    print(f"Train: {len(new_train)}")
    print(f"Val: {len(val)}")

    test = []
    for item in tqdm(test_data):
        label = MAP_LABELS[item[0]]
        text = item[1]
        doc = nlp(text)
        words = [w.text for w in doc]
        test.append({"id": cnt, "text": text, "label": label, "words": words})
        cnt += 1

    print(f"Test: {len(test)}")

    train_word_cnt = Counter()
    for item in tqdm(new_train, total=len(new_train)):
        train_word_cnt.update(item["words"])

    train_word_cnt = sorted(train_word_cnt.items(), key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE]
    vocab = {word: w_id for w_id, (word, _) in enumerate(train_word_cnt)}
    vocab["<pad>"] = len(vocab)
    vocab["<unk>"] = len(vocab)

    torch.save(new_train, "./imdb/train.bin")
    torch.save(val, "./imdb/val.bin")
    torch.save(test, "./imdb/test.bin")
    json.dump(vocab, open("./imdb/vocab.json", "w"))


if __name__ == '__main__':
    main()
