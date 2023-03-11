import json

import torch
from torch.utils.data import Dataset

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class TextDataset(Dataset):
    def __init__(self, file_path, vocab_file: str, max_seq_length: int = -1, tokenizer=None):
        self.data = torch.load(file_path, map_location="cpu")
        self.vocab = json.load(open(vocab_file, "r"))
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        res = {
            "id": item["id"],
            "label": item["label"],
            "word_ids": list(map(lambda x: self.vocab[x] if x in self.vocab else self.vocab["<unk>"], item["words"]))
        }
        return res


class TextCollator:
    def __init__(self, vocab_file: str):
        self.vocab = json.load(open(vocab_file))
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]

    def __call__(self, batch):
        ids = []
        word_ids = []
        labels = []
        for b in batch:
            ids.append(b["id"])
            word_ids.append(b["word_ids"])
            labels.append(b["label"])

        labels = torch.tensor(labels, dtype=torch.long)

        max_seq_len = max(map(len, word_ids))
        input_ids = torch.zeros(len(word_ids), max_seq_len, dtype=torch.long).fill_(self.pad_token_id)
        input_mask = torch.zeros(len(word_ids), max_seq_len, dtype=torch.long)
        for b, b_word_ids in enumerate(word_ids):
            input_ids[b, :len(b_word_ids)] = torch.tensor(b_word_ids, dtype=torch.long)
            input_mask[b, :len(b_word_ids)] = 1

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "labels": labels,
            "meta_data": {
                "index": ids
            }
        }
