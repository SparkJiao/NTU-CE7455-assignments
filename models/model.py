import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from general_util.mixin import LogMixin
from general_util.logger import get_child_logger
from modules import layers
from gensim.models.keyedvectors import KeyedVectors

logger = get_child_logger(__name__)


def load_word2vec(vocab_file: str, embedding_path: str, embedding_dim: int):
    logger.info(f"Loading word2vec embedding using KeyedVectors...")
    model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    vocab = json.load(open(vocab_file, "r"))

    weight = torch.Tensor(len(vocab), embedding_dim)
    weight.data.normal_(mean=0.0, std=0.02)
    weight.data[vocab["<pad>"]].zero_()
    for w, w_id in vocab.items():
        if model.has_index_for(w):
            weight[w_id] = torch.as_tensor(model[w], dtype=torch.float)

    return weight


class RNNEncoder(nn.Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0., bidirectional: bool = False):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.output_size = hidden_size if not bidirectional else 2 * hidden_size
        if mode == "rnn":
            self.rnn = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=True)
        elif mode == "lstm":
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               batch_first=True)
        else:
            raise ValueError(mode)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        packed_seq = pack_padded_sequence(hidden_states, mask.sum(dim=1), batch_first=True)
        if self.mode == "rnn":
            output, h_n = self.rnn(packed_seq)
        elif self.mode == "lstm":
            output, (h_n, _) = self.rnn(packed_seq)
        else:
            raise ValueError(self.mode)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, h_n

    def get_output_size(self):
        return self.output_size


class TextClassifier(nn.Module, LogMixin):
    def __init__(self, encoder: nn.Module, vocab_file: str, embedding_dim, output_dim, pretrained_embedding: torch.Tensor = None):
        super().__init__()
        self.vocab = json.load(open(vocab_file, "r"))
        self.vocab_size = len(self.vocab)

        self.encoder = encoder
        self.fc = nn.Linear(self.encoder.get_output_size, output_dim)
        self.apply(self._init_weights)

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False, padding_idx=self.vocab["<pad>"])
        else:
            self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=self.vocab["<pad>"])
            self._init_weights(self.embedding)

        self.init_metric("acc", "loss")

    @staticmethod
    def _init_weights(module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor = None):
        # text = [sent len, batch size]

        embedded = self.embedding(input_ids)
        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.encoder(embedded, mask)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        logits = self.fc(hidden.squeeze(0))
        loss = 0.
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            acc, true_label_num = layers.get_accuracy(logits, labels)
            self.eval_metrics.update("loss", val=loss, n=true_label_num)
            self.eval_metrics.update("acc", val=loss, n=true_label_num)

        return {
            "logits": logits,
            "loss": loss,
        }
