import json
import numpy
import torch

# 载入json文件
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# 存储json文件
def store_json(filename, json_data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=1)

# 排序标注文件的item
def sort_ann_items(ann_items):
    ann_items.sort(key=lambda x: x['doc_id'])
    return ann_items

# 获取标注内容
def get_doc_ann(ann_file_name):
    ann_list = load_json(ann_file_name)
    doc_ann = {}
    for ann in ann_list:
        doc_ann[ann['doc_id']] = []
    for ann in ann_list:
        doc_ann[ann['doc_id']].append((ann['start_sent_idx'], ann['end_sent_idx']))
    return doc_ann

# 获取文档信息
def get_doc_list_info(doc_list, max_seq_lens):
    doc_ids = []
    event_ids = []
    descriptors = []
    contents = []
    valid_lens = []
    for doc in doc_list:
        Descriptor = doc['Descriptor']
        Doc = doc['Doc']
        content = Doc['content']
        content_text = [c['sent_text'] for c in content]
        padding_context = [''] * (max_seq_lens - len(content))
        content_text[len(content):] = padding_context
        doc_ids.append(Doc['doc_id'])
        event_ids.append(Descriptor['event_id'])
        descriptors.append(Descriptor['text'])
        contents.append(content_text)
        valid_lens.append(len(content))
    return doc_ids, event_ids, descriptors, contents, valid_lens

# 获取训练集/验证集信息
def train_dev_data(doc_file, ann_file, max_seq_lens):
    start_anns = []
    end_anns = []
    doc_ann = get_doc_ann(ann_file)
    doc_list = load_json(doc_file)
    for doc in doc_list:
        start_ann = numpy.array([0] * max_seq_lens)
        end_ann = numpy.array([0] * max_seq_lens)
        ann_list = doc_ann.get(doc['Doc']['doc_id'])
        if ann_list is not None:
            for pair in ann_list:
                start_ann[pair[0]] = 1
                end_ann[pair[1]] = 1
        start_anns.append(start_ann)
        end_anns.append(end_ann)
    return get_doc_list_info(doc_list, max_seq_lens), start_anns, end_anns

# 获取测试集信息
def test_data(doc_file, max_seq_lens):
    doc_list = load_json(doc_file)
    return get_doc_list_info(doc_list, max_seq_lens)

# 模型输出的start和end转化为(start_sent_idx, end_sent_idx)对，支持batch
def get_ann_pair(start_list_batch, end_list_batch, valid_lens):
    sum_infos = []
    res = []
    for start_list, end_list, valid_len in zip(start_list_batch, end_list_batch, valid_lens):
        sum_infos.append([(start_list[i]+end_list[i]).item() for i in range(valid_len)])
    #print(sum_infos)
    for arr in sum_infos:
        doc_pairs = []
        idx = 0
        while idx < len(arr):
            if arr[idx] == 2:
                doc_pairs.append((idx, idx))
            elif arr[idx] == 1:
                start_sent_idx = idx
                while True:
                    idx += 1
                    if idx >= len(arr):
                        break
                    if arr[idx] == 1:
                        break
                doc_pairs.append((start_sent_idx, idx))
            idx += 1
        res.append(doc_pairs)
    return res

# 生成每个batch对应标注文件的items
def generate_ann_items(doc_ids, event_ids, start_list_batch, end_list_batch, valid_lens):
    res = []
    ann_pairs_batch = get_ann_pair(start_list_batch, end_list_batch, valid_lens)
    for doc_id, event_id, ann_pairs in zip(doc_ids, event_ids, ann_pairs_batch):
        for ann_pair in ann_pairs:
            datarow = {
                'event_id': event_id.item(),
                'doc_id': doc_id.item(),
                'start_sent_idx': ann_pair[0],
                'end_sent_idx': ann_pair[1],
                'argument': '' ##### to be modified
            }
            res.append(datarow)
    return res


if __name__ == '__main__':
    #train_doc_list_info, start_anns, end_anns = train_dev_data("./ECOM2022/data/train.doc.json", "./ECOM2022/data/train.ann.json", 38)
    #test_doc_list_info = test_data("./ECOM2022/data/test.doc.json", 38)
    #print(train_doc_list_info[0], start_anns[0], end_anns[0])
    #print(test_doc_list_info[0])
    start_list_batch = torch.tensor([[1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0]])
    end_list_batch   = torch.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1, 0]])
    valid_lens = [5, 6]
    print(get_ann_pair(start_list_batch, end_list_batch, valid_lens))

