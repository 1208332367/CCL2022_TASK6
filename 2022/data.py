import json
import numpy
import  torch

def get_doc_ann(ann_file_name):
    with open(ann_file_name) as f:
        ann_list = json.load(f)
        doc_ann = {}
        for ann in ann_list:
            doc_ann[ann['doc_id']] = []
        for ann in ann_list:
            doc_ann[ann['doc_id']].append((ann['start_sent_idx'], ann['end_sent_idx']))
        return doc_ann


def train_dev(doc_file, ann_file, max_seq_lens):
    contents = []
    valid_lens = []
    anns = []
    descriptors = []
    doc_ann = get_doc_ann(ann_file)
    with open(doc_file) as f_doc:
        doc_list = json.load(f_doc)
        for doc in doc_list:
            Descriptor = doc['Descriptor']
            text = Descriptor['text']
            Doc = doc['Doc']
            doc_id = Doc['doc_id']
            # print(doc_id)
            content = Doc['content']
            content_text = [c['sent_text'] for c in content]
            padding_context = [''] * (max_seq_lens - len(content))
            content_text[len(content):] = padding_context
            ann = numpy.array([0] * len(content_text))
            ann_list = doc_ann.get(doc_id)
            if ann_list is not None:
                for a in ann_list:
                    ann[a[0]] = 1
                    ann[a[0] + 1:a[1] + 1] = 2
            valid_lens.append(len(content))
            contents.append(content_text)
            anns.append(ann)
            descriptors.append(text)
    return descriptors, contents, anns, valid_lens,


descriptors, contents, anns, valid_lens = train_dev("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json", 40)
print(descriptors[0])
print(contents[0])
print(anns[0])
print(valid_lens[0])
# # print(torch.Tensor(contents))
# print(torch.Tensor(anns[0]))
