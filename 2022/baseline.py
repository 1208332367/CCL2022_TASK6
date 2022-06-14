import numpy

from dataset import ECOMDatasets
from torch.utils.data import DataLoader
from model import Model
import torch
from MaskedBECLoss import MaskBECLoss

datasets = ECOMDatasets("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json")
dev_datasets = ECOMDatasets("./ECOM2022/data/dev.doc.json", "./ECOM2022/data/dev.ann.json")
dataloader = DataLoader(datasets, batch_size=2, shuffle=True)
dev_dataloader = DataLoader(dev_datasets, batch_size=16, shuffle=False)
model = Model(hidden_states=256, num_layers=2, dropout=0.3,
              bidirectional=True, sentence_max_length=512, descriptor_max_length=20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# for data in dataloader:
#     print(data)
#     #  descriptors, contents, start_anns, end_anns, valid_lens[idx], idx
#     break
# with torch.no_grad():
#     start, end = model(data[0], data[1], device)
# print(start.shape, end.shape)

# start和end的shape: torch.Size([2, 38, 2])   batch_size, seq_lens, target_label(0,1)

num_epochs = 32
lr = 0.2
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(num_epochs):
    print("epoch:", epoch + 1)
    model.train()
    for i, (descriptors, contents, start_anns, end_anns, valid_lens, _) in enumerate(dataloader):
        # use gpu
        start_anns = start_anns.type(torch.float32).to(device)
        end_anns = end_anns.type(torch.float32).to(device)
        optimizer.zero_grad()
        start, end = model(descriptors, contents, device)
        l = MaskBECLoss(start.squeeze(-1), start_anns, valid_lens) + \
            MaskBECLoss(end.squeeze(-1), end_anns, valid_lens)
        if i % 10 == 0:
            print("loss: ", l)
        l.backward()
        optimizer.step()
        break
    model.eval()
    total = 0
    acc = 0
    for i, (descriptors, contents, start_anns, end_anns, valid_lens, _) in enumerate(dataloader):
        start_anns = start_anns.to(device).type(torch.float32)
        end_anns = end_anns.to(device).type(torch.float32)
        start, end = model(descriptors, contents, device)
        start_dev = [i[:v].equal(j[:v]) for i, j, v in zip(start.squeeze(-1), start_anns, valid_lens)]
        end_dev = [i[:v].equal(j[:v]) for i, j, v in zip(end.squeeze(-1), end_anns)]
        static = [i & j for i, j in zip(start_dev, end_dev)]
        total += len(static)
        acc += torch.tensor(numpy.array(static)).sum()
        break
    print('acc:', acc / total)
