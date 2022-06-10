import torch
import torch.nn as nn
from torch.optim import Adam

import transformers
from transformers import BertTokenizer
from transformers import AutoModel
from dataset import ECOMDatasets
from torch.utils.data import DataLoader
from model import Model

datasets = ECOMDatasets("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json")
print(datasets[0])
dataloader = DataLoader(datasets, batch_size=2, shuffle=True)
# print(datasets[0])
# print(datasets[0][1].shape)
# model = Model(hidden_states=256, output_states=3, num_layers=2, bidirectional=True)

i = 0
for batch in dataloader:
    i += 1
    print(batch)
    if i > 5:
        break

# num_epochs = 32
# lr = 0.05
# optimizer = Adam(model.parameters(), lr=lr)
# loss = nn.CrossEntropyLoss()
# for epoch in range(num_epochs):
#     for data in datasets:
#         optimizer.zero_grad()
#         out = model(data[0]).squeeze()
#         labels = torch.from_numpy(data[1])
#         l = loss(out.squeeze(), labels)
#         print(l)
#         l.backward()
#         optimizer.step()
