import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig


class Model(nn.Module):
    def __init__(self, hidden_states, output_states, num_layers, bidirectional):
        super().__init__()
        self.model = AutoModel.from_pretrained('bert-base-chinese')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.config = AutoConfig.from_pretrained('bert-base-chinese')
        self.gru = nn.GRU(self.config.hidden_size, hidden_states, num_layers, bidirectional)
        self.lin = nn.Linear(hidden_states, output_states)

    def forward(self, sentences):
        # [batch_size, sentences_seqs, sequence_lens]
        tokens = self.tokenizer(sentences, padding=True, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
        # print(input_ids, token_type_ids, attention_mask)
        # [batch_size, sentences_seqs, sequence_max_lens, 748]
        out = self.model(input_ids, token_type_ids, attention_mask).pooler_output
        # print(out.shape)
        out = out.unsqueeze(0)
        # print(out.shape)
        # [batch_size, sentences_seqs, 748]
        # out = out.sum(dim=-2)
        # [batch_size, sentences_seqs, hidden_states] , [batch_size, hidden_states]
        output, h_n = self.gru(out)
        output = self.lin(output)
        return output


