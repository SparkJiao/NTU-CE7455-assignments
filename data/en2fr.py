import argparse
import json
import re
import unicodedata

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Seq2SeqDataset(Dataset):
    def __init__(self, file_path, vocab_file: str, max_seq_length: int = -1, tokenizer=None):
        data = json.load(open(file_path, "r"))
        self.inputs = data["inputs"]
        self.outputs = data["outputs"]
        vocab = json.load(open(vocab_file, "r"))
        self.word2index = vocab["word2index"]
        self.index2word = vocab["index2word"]
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        item = {
            "input": self.inputs[index],
            "output": self.outputs[index]
        }
        res = {
            "id": index,
            "input": list(map(lambda x: self.word2index[x] if x in self.word2index else self.word2index["<unk>"],
                              item["input"].split(" "))) + [self.word2index["EOS"]],
            "output": [self.word2index["SOS"]] + list(
                map(lambda x: self.word2index[x] if x in self.word2index else self.word2index["<unk>"],
                    item["output"].split(" "))) + [self.word2index["EOS"]],
            "source": item["input"],
            "target": item["output"],
        }
        return res


class Seq2SeqCollator:
    def __init__(self, vocab_file: str):
        vocab = json.load(open(vocab_file))
        self.word2index = vocab["word2index"]
        self.index2word = vocab["index2word"]

    def __call__(self, batch):
        ids = []
        inputs = []
        outputs = []
        for b in batch:
            ids.append(b["id"])
            inputs.append(b["input"])
            outputs.append(b["output"])

        max_seq_len = max(map(len, inputs))
        input_ids = torch.zeros(len(inputs), max_seq_len, dtype=torch.long).fill_(self.word2index["EOS"])
        input_mask = torch.zeros(len(inputs), max_seq_len, dtype=torch.long)
        for b, b_word_ids in enumerate(inputs):
            input_ids[b, :len(b_word_ids)] = torch.tensor(b_word_ids, dtype=torch.long)
            input_mask[b, :len(b_word_ids)] = 1

        max_seq_len = max(map(len, outputs))
        output_ids = torch.zeros(len(outputs), max_seq_len, dtype=torch.long).fill_(self.word2index["EOS"])
        output_mask = torch.zeros(len(outputs), max_seq_len, dtype=torch.long)
        for b, b_word_ids in enumerate(outputs):
            output_ids[b, :len(b_word_ids)] = torch.tensor(b_word_ids, dtype=torch.long)
            output_mask[b, :len(b_word_ids)] = 1

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "output_ids": output_ids,
            "output_mask": output_mask,
            "meta_data": {
                "input": [b["source"] for b in batch],
                "output": [b["target"] for b in batch],
                "index": ids
            }
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    word2index = {"SOS": 0, "EOS": 1}
    word2count = {}
    index2word = {0: "SOS", 1: "EOS"}
    n_words = 2  # Count SOS and EOS
    inputs = []
    outputs = []

    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
    for pair in pairs:
        inputs.append(pair[1])
        outputs.append(pair[0])

        for word in pair[1].split(' '):
            if word not in word2index:
                word2index[word] = n_words
                word2count[word] = 1
                index2word[n_words] = word
                n_words += 1
            else:
                word2count[word] += 1

        for word in pair[0].split(' '):
            if word not in word2index:
                word2index[word] = n_words
                word2count[word] = 1
                index2word[n_words] = word
                n_words += 1
            else:
                word2count[word] += 1

    # Split the data into train, val and test set
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2,
                                                                              random_state=args.seed)
    val_inputs, test_inputs, val_outputs, test_outputs = train_test_split(test_inputs, test_outputs, test_size=0.5,
                                                                          random_state=args.seed)

    json.dump({"inputs": train_inputs, "outputs": train_outputs}, open(args.output_dir + "/train.json", "w"))
    json.dump({"inputs": val_inputs, "outputs": val_outputs}, open(args.output_dir + "/val.json", "w"))
    json.dump({"inputs": test_inputs, "outputs": test_outputs}, open(args.output_dir + "/test.json", "w"))
    json.dump({"word2index": word2index, "word2count": word2count, "index2word": index2word, "n_words": n_words},
              open(args.output_dir + "/vocab.json", "w"))


if __name__ == '__main__':
    main()
