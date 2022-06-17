from torch.utils.data import Dataset
import dataprocess
from transformers import BertTokenizer


class ECOMDatasets(Dataset):
    def __init__(self, doc_file, ann_file=None, max_seq_lens=38):
        super().__init__()
        self.ann_file = ann_file
        if ann_file is not None:
            doc_list_info, self.start_anns, self.end_anns = dataprocess.train_dev_data(doc_file, ann_file, max_seq_lens)
        else:
            doc_list_info = dataprocess.test_data(doc_file, max_seq_lens)
        self.get_doc_list_info(doc_list_info)

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        if self.ann_file is not None :
            return self.doc_ids[idx], self.event_ids[idx], self.descriptors[idx], self.contents[idx], self.valid_lens[idx], self.start_anns[idx], self.end_anns[idx], idx
        else:
            return self.doc_ids[idx], self.event_ids[idx], self.descriptors[idx], self.contents[idx], self.valid_lens[idx], idx

    def get_doc_list_info(self, doc_list_info):
        self.doc_ids = doc_list_info[0]
        self.event_ids = doc_list_info[1]
        self.descriptors = doc_list_info[2]
        self.contents = doc_list_info[3]
        self.valid_lens = doc_list_info[4]

if __name__ == '__main__':
    train_datasets = ECOMDatasets("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json")
    test_datasets = ECOMDatasets("./ECOM2022/data/test.doc.json")
    print(train_datasets[0])
    #print(test_datasets[0])
