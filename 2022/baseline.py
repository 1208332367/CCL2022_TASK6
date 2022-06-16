import numpy
from tqdm import tqdm
from dataset import ECOMDatasets
from torch.utils.data import DataLoader
from model import Model
import torch
from MaskedBECLoss import MaskBECLoss, toOnehot

datasets = ECOMDatasets("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json")
dev_datasets = ECOMDatasets("./ECOM2022/data/dev.doc.json", "./ECOM2022/data/dev.ann.json")
dataloader = DataLoader(datasets, batch_size=2, shuffle=True)
dev_dataloader = DataLoader(dev_datasets, batch_size=2, shuffle=False)
model = Model(hidden_states=256, num_layers=2, dropout=0.3, sentence_max_length=512, descriptor_max_length=20)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# for name, param in model.named_parameters():
#     print(name, param.size())
#
# unfreeze_layers = ['LM.encoder.layer.10', 'LM.encoder.layer.11', 'LM.pooler', 'gru.', 'lin1.', 'lin2.']
# for name, param in model.named_parameters():
#     param.requires_grad = False
#     for ele in unfreeze_layers:
#         if ele in name:
#             param.requires_grad = True
#             break
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.size())
model.to(device)
# for data in dataloader:
#     print(data)
#     #  descriptors, contents, start_anns, end_anns, valid_lens[idx], idx
#     break
# start, end = model(data[0], data[1], device)
# print(start.shape, end.shape)

# start和end的shape: torch.Size([2, 38, 2])   batch_size, seq_lens, target_label(0,1)

num_epochs = 10
lr = 0.05
threshold = 0.8
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
fout = open('log.out', 'w', encoding='utf-8')
for epoch in range(num_epochs):
    print("epoch:", epoch + 1)
    for i, (descriptors, contents, start_anns, end_anns, valid_lens, _) in enumerate(tqdm(dataloader)):
        # use gpu
        start_anns = start_anns.type(torch.float32).to(device)
        end_anns = end_anns.type(torch.float32).to(device)
        valid_lens = valid_lens.to(device)
        optimizer.zero_grad()
        start, end = model(descriptors, contents, device)
        l = MaskBECLoss(start.squeeze(-1), start_anns, valid_lens) + \
            MaskBECLoss(end.squeeze(-1), end_anns, valid_lens)
        if i % 100 == 0:
            print("loss: ", l.item())
            fout.write('loss: ' + str(l.item()) + '\n')
        l.backward()
        optimizer.step()
    total = 0
    acc = 0
    for i, (descriptors, contents, start_anns, end_anns, valid_lens, _) in enumerate(tqdm(dataloader)):
        start_anns = start_anns.to(device).type(torch.float32)
        end_anns = end_anns.to(device).type(torch.float32)
        with torch.no_grad():
            start, end = model(descriptors, contents, device)
        start = toOnehot(start.squeeze(-1), threshold)
        end = toOnehot(end.squeeze(-1), threshold)
        start_dev = [i[:v].equal(j[:v]) for i, j, v in zip(start, start_anns, valid_lens)]
        end_dev = [i[:v].equal(j[:v]) for i, j, v in zip(end, end_anns, valid_lens)]
        static = [i & j for i, j in zip(start_dev, end_dev)]
        total += len(static)
        acc += torch.tensor(numpy.array(static)).sum()
    print('acc:', acc / total)
    fout.write('acc: ' + str(acc / total) + '\n')
fout.close()