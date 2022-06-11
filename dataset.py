from torch.utils.data import Dataset
import preprocess
from transformers import BertTokenizer


class ECOMDatasets(Dataset):
    def __init__(self, doc_file, ann_file=None, max_seq_lens=38):
        super().__init__()
        self.descriptors, self.contents, self.start_anns, self.end_anns, self.valid_lens = preprocess.train_dev(doc_file, ann_file, max_seq_lens)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        return self.descriptors[idx], self.contents[idx], self.start_anns[idx], self.end_anns[idx], self.valid_lens[idx], idx


# datasets = ECOMDatasets("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json")
# contents, anns = datasets[40:50]
# print(contents, anns)
