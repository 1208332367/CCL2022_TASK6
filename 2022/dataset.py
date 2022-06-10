from torch.utils.data import Dataset
import data
from transformers import BertTokenizer


class ECOMDatasets(Dataset):
    def __init__(self, doc_file, ann_file=None, max_seq_lens=50):
        super().__init__()
        self.contents, self.anns , self.valid_lens = data.train_dev(doc_file, ann_file, max_seq_lens)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        return self.contents[idx], self.anns[idx], self.valid_lens[idx], idx,


# datasets = ECOMDatasets("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json")
# contents, anns = datasets[40:50]
# print(contents, anns)
