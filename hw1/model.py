from typing import Dict, List

import torch
from torch.nn import Embedding
from torch.nn import functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        att: bool,
        att_unit: int,
        att_hops: int
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.lstm = torch.nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidirectional, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.encoder_output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder_output_size, self.num_class)
        )
        self.att = att
        if self.att:
            self.att_unit = att_unit
            self.att_hops = att_hops
            self.w1 = torch.nn.Linear(self.encoder_output_size, self.att_unit)
            self.w2 = torch.nn.Linear(self.att_unit, self.att_hops)
            self.fc = torch.nn.Linear(self.att_hops * self.encoder_output_size, self.encoder_output_size)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def _get_mask(self, batch_lens, shape):
        mask = torch.zeros(shape)
        batch_idx = torch.cat([torch.full((shape[2] - l,), i) for i, l in enumerate(batch_lens)])
        seq_idx = torch.cat([torch.arange(l, shape[2]).long() for l in batch_lens])
        mask[batch_idx, :, seq_idx] = -float('inf')
        return mask

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output_dict: Dict[str, torch.Tensor] = {}
        x, y = batch['text'], batch['intent']
        batch_size = x.size(0)
        # x.shape = [batch_size, max_len]
        x = self.embed(x)
        # x.shape = [batch_size, max_len, embed_dim]
        x, _ = self.lstm(x)
        # x.shape = [batch_size, max_len, hid_dim])
        if self.att:
            A = self.w2(torch.tanh(self.w1(x)))
            # A.shape = [batch_size, max_len, r]
            A = A.permute(0, 2, 1)
            mask = self._get_mask(batch['len'], A.shape)
            A = A + mask.type_as(A)
            A = A.softmax(-1)
            # A.shape = [batch_size, r, max_len]
            x = torch.bmm(A,x)
            # A.shape = [batch_size, r, hid_dim]
            x = x.reshape(batch_size, -1)
            x = self.fc(x)
        else:
            x = x[torch.arange(0, len(batch['len'])), batch['len']-1]
        # x.shape = [batch_size, hid_dim])

        pred_logits = self.classifier(x)
        output_dict['pred_logits'] = pred_logits
        output_dict['pred_labels'] = pred_logits.max(1, keepdim=True)[1].reshape(-1)
        output_dict['loss'] = F.cross_entropy(pred_logits, y.long())

        return output_dict

class SeqTagging(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        num_cnn_layers: int,
        hidden_size: int,
        num_rnn_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        class_weight: torch.tensor=None
    ) -> None:
        super(SeqTagging, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.num_cnn_layers = num_cnn_layers
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.class_weight = class_weight
        
        cnn = []
        for i in range(num_cnn_layers):
            conv_layer = torch.nn.Sequential(
                ConvNorm(self.embed_dim,
                         self.embed_dim,
                         kernel_size=3, stride=1,
                         padding=None, dilation=1, 
                         bias=False, nonlinearity='relu'),
                torch.nn.BatchNorm1d(self.embed_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout()
            )
            cnn.append(conv_layer)
        self.cnn = torch.nn.ModuleList(cnn)

        if self.num_rnn_layers > 0:
            self.rnn = torch.nn.LSTM(self.embed_dim, self.hidden_size, self.num_rnn_layers, dropout=dropout, bidirectional=self.bidirectional)

        self.tag_lassifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder_output_size, self.num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.num_rnn_layers <= 0:
            return self.embed_dim
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def _get_idx(self, tokens_len) -> (List[int], List[int]):
        batch_idx = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(tokens_len)])
        tok_idx = torch.cat([torch.arange(0, t_len) for t_len in tokens_len])
        return batch_idx, tok_idx

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output_dict: Dict[str, torch.Tensor] = {}
        x, y = batch['tokens'], batch['tags']
        # x.shape = [batch_size, max_len], y.shape = List[tensor]
        x = self.embed(x)
        # x.shape = [batch_size, max_len, embed_dim]
        x = x.permute(0, 2, 1)
        # x.shape = [batch_size, embed_dim, max_len]
        for conv in self.cnn:
            x = conv(x) + x
        x = x.permute(2, 0, 1)
        # x.shape = [max_len, batch_size, embed_dim]
        if self.num_rnn_layers > 0:
            self.rnn.flatten_parameters()
            x, _ = self.rnn(x)
        # x.shape = [max_len, batch_size, hid_dim])
        x = x.permute(1, 0, 2)
        # x.shape = [batch_size, max_len, hid_dim])
        pred_logits = self.tag_lassifier(x)
        # pred_logits.shape = [batch_size, max_len, num_class])
        idx = self._get_idx(batch['len'])
        pred_logits = pred_logits[idx]
        output_dict['loss'] = F.cross_entropy(pred_logits, torch.cat(y).long(), weight=self.class_weight)

        pred_labels = pred_logits.max(-1, keepdim=True)[1].reshape(-1)
        output_dict['pred_logits'] = pred_logits.split(batch['len'].tolist())
        output_dict['pred_labels'] = pred_labels.split(batch['len'].tolist())

        return output_dict

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, nonlinearity='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.kaiming_normal_(
            self.conv.weight, mode='fan_out', nonlinearity=nonlinearity)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal