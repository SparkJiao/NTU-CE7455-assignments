import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from general_util.mixin import LogMixin
from general_util.logger import get_child_logger
from modules import layers
from typing import List, Union
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
            weight[w_id] = torch.tensor(model[w], dtype=torch.float)

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
        bsz = hidden_states.size(0)
        packed_seq = pack_padded_sequence(hidden_states, mask.sum(dim=1).to(device="cpu"), batch_first=True, enforce_sorted=False)
        if self.mode == "rnn":
            output, h_n = self.rnn(packed_seq)
        elif self.mode == "lstm":
            output, (h_n, _) = self.rnn(packed_seq)
        else:
            raise ValueError(self.mode)
        h_n = h_n.transpose(0, 1).reshape(bsz, -1)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, h_n

    def get_output_size(self):
        return self.output_size


class MLPEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: Union[int, List[int]], num_layers: int, dropout: float = 0.0):
        super().__init__()
        fc_layers = []
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_layers
        layer_input_size = input_size
        for i in range(num_layers):
            fc_layers.append(nn.Linear(layer_input_size, hidden_size[i]))
            fc_layers.append(nn.GELU())
            if dropout:
                fc_layers.append(nn.Dropout(p=dropout))
            layer_input_size = hidden_size[i]
        self.layers = nn.Sequential(*fc_layers)
        self.output_size = hidden_size[-1]

    def get_output_size(self):
        return self.output_size

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor):
        lengths = input_mask.sum(dim=1)
        hidden_states = hidden_states * input_mask.unsqueeze(-1)
        hidden_states = hidden_states.sum(dim=1) / lengths.unsqueeze(1).to(dtype=hidden_states.dtype)
        outputs = self.layers(hidden_states)
        return outputs


class CNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size: List[int], dropout: float = 0.0):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size[i]),
                nn.ReLU(),
            ) for i in range(len(kernel_size))
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_size = len(kernel_size) * hidden_size

    @staticmethod
    def conv_and_pool(x, conv):
        x = conv(x)
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def get_output_size(self):
        return self.output_size

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor):
        hidden_states = hidden_states.transpose(1, 2).contiguous()

        outputs = []
        for conv in self.conv:
            x = self.conv_and_pool(hidden_states, conv)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        return self.dropout(x)


class TextClassifier(nn.Module, LogMixin):
    def __init__(self, encoder: nn.Module, vocab_file: str, embedding_dim, output_dim, pretrained_embedding: torch.Tensor = None):
        super().__init__()
        self.vocab = json.load(open(vocab_file, "r"))
        self.vocab_size = len(self.vocab)

        self.encoder = encoder
        self.fc = nn.Linear(self.encoder.get_output_size(), output_dim)
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

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor, labels: torch.Tensor = None):
        # text = [sent len, batch size]

        embedded = self.embedding(input_ids)
        # embedded = [sent len, batch size, emb dim]

        res = self.encoder(embedded, input_mask)
        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        if isinstance(res, tuple):
            output, hidden = res
        else:
            hidden = res

        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        logits = self.fc(hidden)
        loss = 0.
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            acc, true_label_num = layers.get_accuracy(logits, labels)
            self.eval_metrics.update("loss", val=loss, n=true_label_num)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)

        return {
            "logits": logits,
            "loss": loss,
        }
