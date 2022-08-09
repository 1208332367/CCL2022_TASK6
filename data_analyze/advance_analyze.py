import json, os
import numpy
import jieba
from ltp import LTP
import jieba

ltp = LTP(path='base')

data_root = '../data/ECOB-ZH'

train_doc_file = os.path.join(data_root, 'train.doc.json')
dev_doc_file = os.path.join(data_root, 'dev.doc.json')
test_doc_file = os.path.join(data_root, 'test.doc.json')

train_ann_file = os.path.join(data_root, 'train.ann.json')
dev_ann_file = os.path.join(data_root, 'dev.ann.json')

pred_file = os.path.join('result/chinese_result', 'mrc.ann.json')

output_file = os.path.join(data_root, 'advance_analyze.json')

# 载入json文件
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# 存储json文件
def store_json(filename, json_data, json_indent=1):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=json_indent, ensure_ascii=False)

# 排序标注文件的item
def sort_ann_items(ann_items):
    ann_items.sort(key=lambda x: x['doc_id'])
    return ann_items

# 比较两个字典是否部分相等
def compare_two_dict(dict1, dict2, key_list):
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    if len(key_list) != 0:
        for key in key_list:
            if key in keys1 and key in keys2:
                if dict1[key] != dict2[key]:
                    return False
            else:
                raise Exception('key_list contains error key')
        return True
    else:
        raise Exception('key_list is null')

def compare_item(ann_item, predict_item, type='full'):
    key_list = ['event_id', 'doc_id', 'start_sent_idx', 'end_sent_idx', 'argument']
    if type == 'full':
        return compare_two_dict(ann_item, predict_item, key_list)
    if type == 'fragment':
        return compare_two_dict(ann_item, predict_item, key_list[0:4])
    if type == 'argument':
        return compare_two_dict(ann_item, predict_item, key_list[0:2] + key_list[4:5])

# 获取标注内容
def get_doc_ann(ann_file_name):
    ann_list = load_json(ann_file_name)
    doc_ann = {}
    for ann in ann_list:
        doc_ann[ann['doc_id']] = []
    for ann in ann_list:
        doc_ann[ann['doc_id']].append([ann['start_sent_idx'], ann['end_sent_idx'], ann['argument']]) # 0630 wjh add 'argument'
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

# 给task1预测的json增加argument(空串)
def arg_padding(file_in, file_out):
    items = load_json(file_in)
    for item in items:
        if 'argument' not in item.keys():
            item['argument'] = ""
    store_json(file_out, items)

# 根据doc_id保存生成文件的预测和标注文件，便于对比
def analyze_pred(pred_file, gold_file, output_file):
    pred_json = get_doc_ann(pred_file)
    gold_json = get_doc_ann(gold_file)

    res = []
    #print(gold_json)
    for doc_id, gold_pairs in gold_json.items():
        row = {
            'doc_id': doc_id,
            'gold_pairs': [','.join((str(gold_pair[0]), str(gold_pair[1]), gold_pair[2])) for gold_pair in gold_pairs],
            'pred_pairs': [','.join((str(pred_pair[0]), str(pred_pair[1]), pred_pair[2])) for pred_pair in pred_json[doc_id]] if doc_id in pred_json.keys() else [],
        }
        res.append(row)   
    #print(res)
    store_json(output_file, res, json_indent=4)

def itemCount(file):
    pred_json = load_json(file)     
    return len(pred_json)

def sentCharCount(doc_file):
    docs = load_json(doc_file)
    total = 0 
    for doc in docs:
        contents = doc['Doc']['content']
        for content in contents:
            q = content['sent_text']
            total += len(q)
    return total

def errCodeCount(file):
    docs = load_json(file)
    cnt = 0
    for doc in docs:
        contents = doc['Doc']['content']
        for content in contents:
            if content['trans_code'] != 0:
                print(doc)
                cnt += 1
    return cnt

def getSpaceDescriptorCount(file):
    pred_json = load_json(file)  
    cnt = 0
    for item in pred_json: 
        if ' ' in item['argument']:
            print(item['doc_id'], item['argument'])
            cnt += 1
    return cnt

def segCompare(file_in, file_out):
    doc_list = load_json(file_in)
    res = {}
    _, event_ids, descriptors, _, _= get_doc_list_info(doc_list, 38)
    
    for event_id, descriptor in zip(event_ids, descriptors):
        if event_id in res.keys():
            continue
        descriptor = descriptor.strip().replace(' ', ',')
        ltp_segment, _ = ltp.seg([descriptor])
        jieba_segment = list(jieba.cut(descriptor))
        res[event_id] =  {'ltp_seg': '#'.join(ltp_segment[0]).replace(',', ' '), 'jieba_seg': '#'.join(jieba_segment).replace(',', ' ')}
    store_json(file_out, res)

def getDifferentItem(output_file):
    items = load_json(output_file)
    has_different = False
    for item in items:
        if item['gold_pairs'] != item['pred_pairs']:
            print(item)
            has_different = True
    if not has_different:
        print('Completely Same')
    

if __name__ == '__main__':
    '''
    ann_item = {
        "event_id": 0,
        "doc_id": 0,
        "start_sent_idx": 4,
        "end_sent_idx": 5,
        "argument": "无多余新冠疫苗分给印度"
    }
    predict_item = {
        "event_id": 0,
        "doc_id": 0,
        "start_sent_idx": 4,
        "end_sent_idx": 5,
        "argument": "无多余新冠疫苗分给印度"
    }
    print(compare_item(ann_item, predict_item, 'full'))
    '''
    #arg_padding(pred_file, pred_file)
    #analyze_pred(pred_file=pred_file, gold_file=dev_file, output_file=output_file)
    #getDifferentItem(output_file)
    #print(sentCharCount(os.path.join(data_root, 'check2_train_zh_to_en.doc.json')))
    #print(errCodeCount(os.path.join(data_root, 'dev_zh_to_en.doc.json')))
    print(itemCount(pred_file))
    #getSpaceDescriptorCount(pred_file)
    #segCompare(dev_doc_file, output_file)

