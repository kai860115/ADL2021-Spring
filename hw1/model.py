from typing import Dict, List

import torch
import torch.nn as nn
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
        num_classes: int,
        att: bool,
        att_unit: int,
        att_hops: int,
        aux_loss: bool=False
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.att = att

        self.rnn = nn.LSTM(self.embed_dim, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=self.bidirectional, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_output_size, self.num_classes)
        )
        if self.att:
            self.att_unit = att_unit
            self.att_hops = att_hops
            self.attention = Attention(self.encoder_output_size, self.att_unit, self.att_hops)

    @property
    def encoder_output_size(self) -> int:
        # calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        output_dict = {}

        x, y = batch['text'], batch['intent'] # [batch_size, max_len]

        # compute mask
        mask = (x.gt(0)).float() # [batch_size, max_len]

        x = self.embed(x) # [batch_size, max_len, embed_dim]

        packed_x = nn.utils.rnn.pack_padded_sequence(x, batch['len'], batch_first=True)
        self.rnn.flatten_parameters()
        x, (h, _) = self.rnn(packed_x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) # [batch_size, max_len, hid_dim]

        if self.bidirectional:
            h = torch.cat((h[-1], h[-2]), axis=-1) 
        else:
            h = h[-1]

        pred_logits = [self.classifier(h)]

        if self.att:
            M, A = self.attention(x, mask[:, :x.size(1)])
            r = torch.tanh(M.sum(1) / self.att_hops) 
            pred_logits += [self.classifier(r)]
            I = torch.eye(self.att_hops).unsqueeze(0).type_as(A)
            output_dict['penalization'] = ((((torch.bmm(A, A.transpose(1, 2)) - I) ** 2).sum(2).sum(1) + 1e-10) ** 0.5).sum() / A.size(0)
            # ||A * A^T - I||^2_F

        output_dict['pred_logits'] = pred_logits
        output_dict['pred_labels'] = pred_logits[-1].max(1, keepdim=True)[1].reshape(-1)

        output_dict['loss'] = F.cross_entropy(pred_logits[-1], y.long())

        if self.aux_loss:
            output_dict['aux_loss'] = 0
            for p in output_dict['pred_logits'][:-1]:
                output_dict['aux_loss'] += F.cross_entropy(p, y.long())

        return output_dict


class Attention(nn.Module):
    def __init__(self, encoder_output_size, att_unit, att_hops):
        super(Attention, self).__init__()
        self.w1 = nn.Linear(encoder_output_size, att_unit, bias=False)
        self.w2 = nn.Linear(att_unit, att_hops, bias=False)
        self.att_unit = att_unit
        self.att_hops = att_hops
        
    def forward(self, x, mask):
        mask = mask.unsqueeze(1) # [batch_size, 1, seq_len]
        mask[mask == 0] = -1e12
        A = self.w2(torch.tanh(self.w1(x))) # [batch, seq_len, att_hops]
        A = A.permute(0, 2, 1)
        A = A + mask
        A = A.softmax(-1)
        M = torch.bmm(A, x)
        return M, A


class SeqTagging(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        num_cnn_layers: int,
        hidden_size: int,
        num_rnn_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int,
        no_crf: bool,
        class_weight: torch.tensor=None
    ) -> None:
        super(SeqTagging, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # model architecture
        self.embed_dim = embeddings.size(1)
        self.num_cnn_layers = num_cnn_layers
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.no_crf = no_crf
        
        cnn = []
        for i in range(num_cnn_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim, 5, 1, 2),
                nn.ReLU(),
                nn.Dropout()
            )
            cnn.append(conv_layer)
        self.cnn = nn.ModuleList(cnn)

        if self.num_rnn_layers > 0:
            self.rnn = nn.LSTM(self.embed_dim, self.hidden_size, self.num_rnn_layers, dropout=dropout, bidirectional=self.bidirectional, batch_first=True)

        if self.no_crf:
            self.tag_lassifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.encoder_output_size, self.num_classes)
            )
        else:
            self.crf = CRF(self.encoder_output_size, self.num_classes, dropout)

    @property
    def encoder_output_size(self) -> int:
        # calculate the output dimension of rnn
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
        output_dict = {}

        x = batch['tokens'] # [batch_size, max_len]
        x = self.embed(x) # [batch_size, max_len, embed_dim]

        # CNN 
        x = x.permute(0, 2, 1) # [batch_size, embed_dim, max_len]
        for conv in self.cnn:
            x = conv(x) 
        x = x.permute(0, 2, 1) # [batch_size, max_len, embed_dim]

        # RNN 
        if self.num_rnn_layers > 0:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, batch['len'], batch_first=True)
            self.rnn.flatten_parameters()
            x, _ = self.rnn(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) # [batch_size, max_len, hid_dim])

        batch['mask'] = batch['mask'][:, :x.size(1)]
        batch['tags'] = batch['tags'][:, :x.size(1)]
        y = batch['tags']

        if self.no_crf:
            pred_logits = self.tag_lassifier(x) # [batch_size, max_len, num_classes])
            
            idx = self._get_idx(batch['len'])
            output_dict['loss'] = F.cross_entropy(pred_logits[idx], y[idx])

            pred_labels = pred_logits.max(-1, keepdim=True)[1]
            
            output_dict['pred_logits'] = pred_logits
            output_dict['pred_labels'] = pred_labels.squeeze(2)
            
        else:
            output_dict['loss'] = self.crf.loss(x, y, batch['mask'])
            output_dict['max_score'], output_dict['pred_labels'] = self.crf(x, batch['mask'])

        return output_dict


def log_sum_exp(x):
    max_score = x.max(-1)[0]
    return max_score + torch.log(torch.sum(torch.exp(x - max_score.unsqueeze(-1)), -1))


class CRF(nn.Module):
    def __init__(self, in_dim, num_classes, dropout):
        super(CRF, self).__init__()
        self.num_classes = num_classes + 2
        self.start_idx = self.num_classes - 2
        self.stop_idx = self.num_classes - 1

        # Maps the output of the LSTM into tag space.
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, self.num_classes)
        )

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.num_classes, self.num_classes), requires_grad=True)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.start_idx, :] = -1e5
        self.transitions.data[:, self.stop_idx] = -1e5

    def forward(self, x, mask):
        x = self.fc(x)
        return self._viterbi_decode(x, mask)
    
    def loss(self, x, tags, mask):
        x = self.fc(x)

        forward_score = self._forward_alg(x, mask)
        gold_score = self._score_sentence(x, tags, mask)
        loss = (forward_score - gold_score).mean()

        return loss

    def _score_sentence(self, x, tags, mask):
        B, L, C = x.shape
        seq_len = mask.sum(-1).long() # [B]

        emit_scores = x.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1) # [B, L]
        
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long).type_as(tags)
        tags = torch.cat([start_tag, tags], 1) # [B, L + 1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]] # [B, L]

        last_tag = tags.gather(dim=1, index=seq_len.unsqueeze(-1)).squeeze(-1)
        last_score = self.transitions[self.stop_idx, last_tag] # [B]

        score = ((emit_scores + trans_scores) * mask).sum(-1) + last_score

        return score # [B]

    @torch.no_grad()
    def _viterbi_decode(self, x, mask):
        B, L, C = x.shape
        seq_len = mask.sum(-1).long() # [B]

        bptrs = torch.zeros_like(x).long() # [B, L, C]

        max_score = torch.full_like(x[:, 0, :], -1e5) # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = mask[:, t].unsqueeze(-1) # [B, 1]
            emit_score_t = x[:, t] # [B, C]

            score_t = max_score.unsqueeze(1) + self.transitions
            score_t, bptrs[:, t, :] = score_t.max(-1)
            score_t += emit_score_t

            max_score = score_t * mask_t + max_score * (1 - mask_t)

        max_score += self.transitions[self.stop_idx]
        path_score, best_tags = max_score.max(-1)
        
        best_paths = []
        for b in range(B):
            best_tag_id = best_tags[b].item()
            best_path = [best_tag_id]

            for bptrs_t in bptrs[b, :seq_len[b]].flip(0):
                best_tag_id = bptrs_t[best_tag_id]
                best_path += [best_tag_id]

            best_path.pop()
            best_path.reverse()
            best_paths.append(best_path + [0] * (L - len(best_path)))
        
        best_paths = torch.tensor(best_paths)

        return max_score, best_paths


    def _forward_alg(self, x, mask):
        B, L, C = x.shape

        scores = torch.full_like(x[:, 0, :], -1e5)
        scores[:, self.start_idx] = 0

        for t in range(L):
            emit_score_t = x[:, t].unsqueeze(-1) # [B, C, 1]
            score_t = log_sum_exp(scores.unsqueeze(1) + self.transitions.unsqueeze(0) + emit_score_t)  # [B, C]

            mask_t = mask[:, t].unsqueeze(-1) # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)

        scores = log_sum_exp(scores + self.transitions[self.stop_idx])

        return scores # [B]