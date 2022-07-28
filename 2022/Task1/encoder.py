import torch
import torch.nn as nn
from transformers import BertModel


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.LM = BertModel.from_pretrained(config.bert)
        self.word_dropout = nn.Dropout(config.dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim=config.embedding_dim, num_heads=2,
                                                     dropout=config.dropout)
        self.sentence_gru = nn.GRU(config.embedding_dim, config.hidden_dim // 2, config.num_layers,
                                   bidirectional=True, dropout=config.dropout)
        self.classifier = nn.Linear(config.embedding_dim + config.hidden_dim, config.label_size)

    def forward(self, sent_tensor: torch.Tensor, event_tensor: torch.Tensor):
        # sentences embedding
        batch_sent = sent_tensor  # (batch_size, max_sent_len, max_num_len)
        batch_sent = batch_sent[:, :, :240]  # (batch_size, max_sent_len, max_num_len)
        batch_sent_flatten = batch_sent.view(-1, batch_sent.shape[2])  # (batch_size*max_sent_len, max_num_len)
        # (batch_size*max_sent_len, max_num_len)
        batch_sent_output = self.LM(batch_sent_flatten, attention_mask=batch_sent_flatten.gt(0))[0]
        # (batch_size*max_sent_len, max_num_len, config.embedding_dim)
        batch_sent_output = batch_sent_output[:, 0, :].view(batch_sent.shape[0], batch_sent.shape[1], -1)
        # (batch_size, max_seq_len, config.embedding_dim)
        batch_sent_output = self.word_dropout(batch_sent_output)
        batch_sent_output = batch_sent_output.permute(1, 0, 2)
        # (max_sent_len, batch_size, config.embedding_dim)
        gru_out, _ = self.sentence_gru(batch_sent_output)
        # (max_sent_len, batch_size, config.hidden_dim)
        gru_out = gru_out.permute(1, 0, 2)

        # event embedding
        batch_event_output = self.LM(event_tensor, attention_mask=event_tensor.gt(0))[0]
        # (batch_size, max_event_len, config.embedding_dim)
        batch_event_output = self.word_dropout(batch_event_output)
        batch_event_output = batch_event_output.permute(1, 0, 2)
        # (max_event_len, batch_size, config.embedding_dim)

        # attention mechanism
        attention_out, _ = self.cross_attention(batch_sent_output, batch_event_output, batch_event_output)
        attention_out = attention_out.permute(1, 0, 2)
        # (max_sent_len, batch_size, config.embedding_dim)
        features = torch.cat((attention_out, gru_out), dim=-1)  # concatenate
        # features = attention_out + gru_out  # add
        # features = gru_out  # baseline
        pred = self.classifier(features)  # (batch_size, max_sent_len, config.label_size)
        return features, pred
