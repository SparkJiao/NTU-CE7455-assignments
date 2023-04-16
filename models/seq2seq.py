import json
import random
from typing import Dict, Callable
from typing import Union

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import LayerNorm
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin

logger = get_child_logger(__name__)


class RNNEncoder(nn.Module):
    def __init__(self, mode: str, vocab,
                 input_size: int, hidden_size: int, num_layers: int, dropout: float = 0., bidirectional: bool = False):
        super().__init__()
        self.mode = mode
        self.vocab = vocab
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab["word2index"]), embedding_dim=input_size)
        self.input_size = input_size
        self.output_size = hidden_size if not bidirectional else 2 * hidden_size
        if mode == "rnn":
            self.rnn = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=True)
        if mode == "gru":
            self.rnn = nn.GRU(input_size=input_size,
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

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        hidden_states = self.embedding(input_ids)
        bsz = hidden_states.size(0)
        packed_seq = pack_padded_sequence(hidden_states, mask.sum(dim=1).to(device="cpu"), batch_first=True, enforce_sorted=False)
        if self.mode == "rnn":
            output, h_n = self.rnn(packed_seq)
        elif self.mode == "gru":
            output, h_n = self.rnn(packed_seq)
        elif self.mode == "lstm":
            output, h_n = self.rnn(packed_seq)
            # print(h_n[0].size(), h_n[1].size())
        else:
            raise ValueError(self.mode)
        # h_n = h_n.transpose(0, 1).reshape(bsz, -1)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, h_n

    def get_output_size(self):
        return self.output_size


class CustomTransformerEncoder(nn.Module):
    def __init__(self, vocab: Dict[str, Dict], num_encoder_layers: int,
                 d_model: int, n_head: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.GELU(), layer_norm_eps: float = 1e-5):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab["word2index"]), embedding_dim=d_model)

        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first=True)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        hidden_states = self.embedding(input_ids)
        output = self.encoder(hidden_states, src_key_padding_mask=~(mask.bool()))
        return output, output[:, :1, :].transpose(0, 1).contiguous()


class Decoder(nn.Module):
    def __init__(self, mode: str, vocab: Dict[str, Dict],
                 input_size: int, hidden_size: int, num_layers: int, dropout: float = 0., bidirectional: bool = False,
                 attention: nn.Module = None):
        super().__init__()
        self.mode = mode
        self.vocab = vocab

        self.input_size = input_size
        self.output_size = hidden_size if not bidirectional else 2 * hidden_size
        if mode == "rnn":
            self.rnn = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=True)
        if mode == "gru":
            self.rnn = nn.GRU(input_size=input_size,
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

        self.embedding = nn.Embedding(len(self.vocab["word2index"]), self.input_size)
        self.attention = attention
        if self.attention is not None:
            self.out = nn.Linear(self.attention.get_output_size() + self.output_size, len(self.vocab["word2index"]))
        else:
            self.out = nn.Linear(self.output_size, len(self.vocab["word2index"]))
        self.d_size = num_layers * 2 if bidirectional else num_layers

    def forward(self, input_ids, encoder_outputs, encoder_mask, hidden=None):
        bsz = input_ids.size(0)
        output = self.embedding(input_ids)
        output = torch.relu(output)
        if self.mode == "rnn":
            output, h_n = self.rnn(output, hidden)
        elif self.mode == "gru":
            if isinstance(hidden, tuple):
                hidden = hidden[0]
                if hidden.size(0) != self.d_size:
                    hidden = hidden.mean(dim=0, keepdim=True).repeat(self.d_size, 1, 1)
            output, h_n = self.rnn(output, hidden)
        elif self.mode == "lstm":
            output, h_n = self.rnn(output, hidden)
        else:
            raise ValueError(self.mode)
        if self.attention is not None:
            attention_output = self.attention(output, encoder_outputs, encoder_mask)
            output = torch.cat([attention_output, output], dim=-1)
        output = self.out(output)
        return output, h_n


class Attention(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, hidden_dim: int):
        super().__init__()
        self.q = nn.Linear(q_dim, hidden_dim)
        self.k = nn.Linear(k_dim, hidden_dim)
        self.v = nn.Linear(v_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, v_dim)
        self.output_size = v_dim

    def forward(self, q, k, mask=None):
        q = self.q(q)
        k = self.k(k)
        v = self.v(k)
        scores = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            expand_mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            scores = scores.masked_fill((1 - expand_mask).bool(), -10000)
        scores = torch.softmax(scores, dim=-1)
        output = torch.bmm(scores, v)
        output = self.out(output)
        return output

    def get_output_size(self):
        return self.output_size


class EncoderDecoder(nn.Module, LogMixin):
    def __init__(self, vocab_file: str, encoder: Union[DictConfig, nn.Module], decoder: Union[DictConfig, nn.Module],
                 teacher_forcing_ratio: float = 0.5, max_output_length: int = 40):
        super().__init__()
        self.vocab = json.load(open(vocab_file, "r"))
        if isinstance(encoder, DictConfig):
            self.encoder = hydra.utils.instantiate(encoder, vocab=self.vocab)
        else:
            self.encoder = encoder
        if isinstance(decoder, DictConfig):
            self.decoder = hydra.utils.instantiate(decoder, vocab=self.vocab)
        else:
            self.decoder = decoder

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_output_length = max_output_length
        self.apply(self._init_weights)
        self.init_metric("loss")

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

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor, output_ids: torch.Tensor, output_mask: torch.Tensor):
        if not self.training:
            return self.generate(input_ids, input_mask)

        encoder_outputs, encoder_hidden = self.encoder(input_ids, input_mask)

        target_length = output_ids.size(1)
        # print(input_ids.size())
        # print(output_ids.size())
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        decoder_outputs = []
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(output_ids[:, di].unsqueeze(1), encoder_outputs, input_mask,
                                                              hidden=encoder_hidden)
                encoder_hidden = decoder_hidden
                decoder_outputs.append(decoder_output)
        else:
            decoder_output, decoder_hidden = self.decoder(output_ids[:, 0].unsqueeze(1), encoder_outputs, input_mask,
                                                          hidden=encoder_hidden)
            encoder_hidden = decoder_hidden
            next_id = decoder_output.argmax(dim=-1)
            decoder_outputs.append(decoder_output)
            for di in range(1, target_length):
                decoder_output, decoder_hidden = self.decoder(next_id, encoder_outputs, input_mask,
                                                              hidden=encoder_hidden)
                encoder_hidden = decoder_hidden
                next_id = decoder_output.argmax(dim=-1)
                decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        shifted_logits = decoder_outputs[:, :-1, :].contiguous()
        shifted_labels = output_ids[:, 1:].contiguous()
        shifted_mask = output_mask[:, 1:].contiguous()
        shifted_labels = shifted_labels.masked_fill((1 - shifted_mask).bool(), -1)
        loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        return {
            "logits": decoder_outputs,
            "loss": loss
        }

    def generate(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        encoder_outputs, encoder_hidden = self.encoder(input_ids, input_mask)
        bsz = input_ids.size(0)
        sos_token_id = self.vocab["word2index"]["SOS"]
        output_ids = torch.tensor([[sos_token_id]], device=input_ids.device).expand(bsz, 1)
        for di in range(self.max_output_length):
            decoder_output, decoder_hidden = self.decoder(output_ids[:, -1].unsqueeze(1), encoder_outputs, input_mask,
                                                          hidden=encoder_hidden)
            encoder_hidden = decoder_hidden
            next_id = decoder_output.argmax(dim=-1)
            output_ids = torch.cat([output_ids, next_id], dim=-1)

        generated_seq = []
        for i in range(bsz):
            generated_seq.append([self.vocab["index2word"][str(idx.item())] for idx in output_ids[i]])
        return {"loss": 0., "generated_seq": generated_seq}
