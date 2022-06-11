import torch
import torch.nn as nn

from dataset import ECOMDatasets
from torch.utils.data import DataLoader
from model import Model

datasets = ECOMDatasets("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json")

dataloader = DataLoader(datasets, batch_size=2, shuffle=True)

model = Model(hidden_states=256, output_states=2, num_layers=2,
              dropout=0.3, bidirectional=True, sentence_max_length=512, descriptor_max_length=20)

for data in dataloader:
    print(data)
    #  descriptors, contents, start_anns, end_anns, valid_lens[idx], idx
    break


with torch.no_grad():
    start, end = model(data[0], data[1])

# start和end的shape: torch.Size([2, 38, 2])   batch_size, seq_lens, target_label(0,1)


# num_epochs = 32
# lr = 0.1
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
