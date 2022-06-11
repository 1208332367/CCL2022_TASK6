import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig


class Model(nn.Module):
    def __init__(self, hidden_states, output_states, num_layers,
                 dropout, bidirectional, sentence_max_length=512, descriptor_max_length=20):
        super().__init__()
        self.sentence_max_length = sentence_max_length
        self.descriptor_max_length = descriptor_max_length
        self.LM = AutoModel.from_pretrained('bert-base-chinese')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.config = AutoConfig.from_pretrained('bert-base-chinese')
        self.gru = nn.GRU(self.config.hidden_size, hidden_states, num_layers, bidirectional, dropout=dropout)
        # self.cross_attention = torch.nn.MultiheadAttention(embed_dim=attention_states, num_heads=num_heads,
        #                                                    dropout=dropout)
        self.lin = nn.Linear(self.config.hidden_size, hidden_states)
        self.event_start_lin = nn.Linear(hidden_states, output_states)
        self.event_end_lin = nn.Linear(hidden_states, output_states)

    def forward(self, descriptor, sentences):
        #  [(batch_size), (batch_size), ... , len(sentences)-1 ] ->  [batch_size * len(sentences)]
        batch_size = len(sentences[0])
        sentences_tokens = self.tokenizer([e for t in sentences for e in t],
                                          add_special_tokens=True,
                                          max_length=self.sentence_max_length,
                                          padding='max_length',
                                          return_tensors='pt')
        # [batch_size * sentences_seqs(38),  sentence_max_length]
        sentences_input_ids = sentences_tokens['input_ids']
        sentences_token_type_ids = sentences_tokens['token_type_ids']
        sentences_attention_mask = sentences_tokens['attention_mask']
        # print(input_ids, token_type_ids, attention_mask)
        sentences_encode = self.LM(sentences_input_ids, sentences_token_type_ids,
                                   sentences_attention_mask).pooler_output
        # [batch_size * sentences_seqs(38), 768]
        sentences_encode = sentences_encode.reshape(batch_size, len(sentences), -1)
        # [batch_size, sentences_seqs(38), 768]
        sentences_encode = sentences_encode.permute(1, 0, 2)
        # [sentences_seqs(38), batch_size, 768]
        gru_output, _ = self.gru(sentences_encode)
        # gru_output -> [sentences_seqs(38), batch_size , hidden_states], h_n-> [batch_size, hidden_states]
        sentences_encode = gru_output.permute(1, 0, 2)
        # [batch_size, sentences_seqs(38), hidden_states]
        descriptor_tokens = self.tokenizer([d for d in descriptor],
                                           add_special_tokens=True,
                                           max_length=self.descriptor_max_length,
                                           padding='max_length',
                                           return_tensors='pt')
        # [batch_size,  descriptor_max_length]
        descriptor_input_ids = descriptor_tokens['input_ids']
        descriptor_token_type_ids = descriptor_tokens['token_type_ids']
        descriptor_attention_mask = descriptor_tokens['attention_mask']
        descriptor_encode = self.LM(descriptor_input_ids, descriptor_token_type_ids,
                                    descriptor_attention_mask).pooler_output
        descriptor_encode = descriptor_encode.unsqueeze(1)
        # [batch_size, 1, 768]
        descriptor_encode = self.lin(descriptor_encode)
        sentences_encode = sentences_encode + descriptor_encode

        # cross-attention
        # attn_output, attn_output_weights = self.cross_attention(gru_output, descriptor_encode, descriptor_encode)
        # [batch_size, sentences_seqs(38), attention_states]

        start = self.event_start_lin(sentences_encode).relu()
        # [batch_size, sentences_seqs(38), 1]
        end = self.event_end_lin(sentences_encode).relu()
        # [batch_size, sentences_seqs(38), 1]
        return start, end