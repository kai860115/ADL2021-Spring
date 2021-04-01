from typing import Dict

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
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.lstm = torch.nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidirectional)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.encoder_output_size, self.num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output_dict: Dict[str, torch.Tensor] = {}
        x, y = batch['text'], batch['intent']
        # print(x.shape, y.shape)
        x = self.embed(x)
        x = x.permute(1, 0, 2)
        # print(x.shape)
        x, _ = self.lstm(x, None)
        # print(x.shape)
        x = x[batch['len'], range(len(batch['len']))]
        # print(x.shape)
        pred_logits = self.classifier(x)
        # print(pred_logits.shape)
        output_dict['pred_logits'] = pred_logits
        # print(pred_logits.max(1, keepdim=True)[1].reshape(-1))
        output_dict['pred_labels'] = pred_logits.max(1, keepdim=True)[1].reshape(-1)
        output_dict['loss'] = F.cross_entropy(pred_logits, y.long())

        return output_dict
