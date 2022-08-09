import os
import json
import argparse

data_root = 'model_combine/task1_combine_data'

item_freq_dict = {}
item_freq_dict_output_file = os.path.join(data_root, 'item_freq_dict.json')

default_combine_list = {
    'attention_k_fold_list_1' :[
        'attention_cat_0.4_512_10fold_fold_0.pred.json',
        'attention_cat_0.4_512_10fold_fold_1.pred.json',
        'attention_cat_0.4_512_10fold_fold_2.pred.json',
        'attention_cat_0.4_512_10fold_fold_3.pred.json',
        'attention_cat_0.4_512_10fold_fold_4.pred.json',
        'attention_cat_0.4_512_10fold_fold_5.pred.json',
        'attention_cat_0.4_512_10fold_fold_6.pred.json',
        'attention_cat_0.4_512_10fold_fold_7.pred.json',
        'attention_cat_0.4_512_10fold_fold_8.pred.json',
        'attention_cat_0.4_512_10fold_fold_9.pred.json'
    ],
    'attention_k_fold_list_2': [
        'attention_cat_0.5_512_10fold_fold_0.pred.json',
        'attention_cat_0.5_512_10fold_fold_1.pred.json',
        'attention_cat_0.5_512_10fold_fold_2.pred.json',
        'attention_cat_0.5_512_10fold_fold_3.pred.json',
        'attention_cat_0.5_512_10fold_fold_4.pred.json',
        'attention_cat_0.5_512_10fold_fold_5.pred.json',
        'attention_cat_0.5_512_10fold_fold_6.pred.json',
        'attention_cat_0.5_512_10fold_fold_7.pred.json',
        'attention_cat_0.5_512_10fold_fold_8.pred.json',
        'attention_cat_0.5_512_10fold_fold_9.pred.json'
    ],
    'attention_k_fold_list_3': [
        'attention_cat_0.4_768_5fold_fold_0.pred.json',
        'attention_cat_0.4_768_5fold_fold_1.pred.json',
        'attention_cat_0.4_768_5fold_fold_2.pred.json',
        'attention_cat_0.4_768_5fold_fold_3.pred.json',
        'attention_cat_0.4_768_5fold_fold_4.pred.json'
    ],
    'final_combine_list': [
        'attention_cat_0.4_512_combine.pred.json',
        'attention_cat_0.5_512_combine.pred.json',
        'attention_cat_0.4_768_combine.pred.json',
        'seq_train_accum2_epoch8.pred.json',
        'seq_enhance_train_accum2_epoch5.pred.json',
        'seq_enhance_train_dev_accum2_epoch8.pred.json'
    ]
}

# 载入json文件
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# 存储json文件
def store_json(filename, json_data, json_indent=1):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=json_indent, ensure_ascii=False)

# 计数+1
def increase_item_cnt(item_str):
    if item_str in item_freq_dict.keys():
        item_freq_dict[item_str] += 1
    else:
        item_freq_dict[item_str] = 1

# 标注item转字符串
def item2str(item):
    return '##'.join((str(item['event_id']), str(item['doc_id']), str(item['start_sent_idx']), str(item['end_sent_idx'])))

# 字符串转标注item
def str2item(item_str):
    res_list = item_str.split('##')
    return {
        "event_id": int(res_list[0]),
        "doc_id": int(res_list[1]),
        "start_sent_idx": int(res_list[2]),
        "end_sent_idx": int(res_list[3]),
        "argument": ""
    }

# 标注文件转字符串集合并计数
def item_list_to_str_set(item_list):
    res = set()
    for item in item_list:
        item_str = item2str(item) 
        increase_item_cnt(item_str) # 计数+1
        res.add(item_str)
    return res

# 字符串集合转标注文件
def str_set_to_item_list(str_list):
    res = []
    for item_str in str_list:
        res.append(str2item(item_str))
    return res

# 标注文件融合
def combine(input_file_list, output_file, select_cnt, item_freq_dict_output_file):
    # 生成item频率
    for input_file in input_file_list:
        str_set = item_list_to_str_set(load_json(input_file)) 
        print('input file [%s] items: %d' %(input_file, len(str_set)))
    store_json(item_freq_dict_output_file, item_freq_dict, json_indent=4)
    # 筛选高频item
    combine_list = []
    for item_str, cnt in item_freq_dict.items():
        if cnt >= select_cnt:
            combine_list.append(str2item(item_str))
    # print(item_freq_dict)
    print('-' * 30)
    print('combine file [%s] items: %d' %(output_file, len(combine_list)))
    store_json(output_file, combine_list, json_indent=4)

def parse_arguments(parser):
    # parameters
    parser.add_argument('--select_cnt', type=int, default=1)
    parser.add_argument('--output_file', type=str, default='combine_test.ann.json')
    parser.add_argument('--combine_file_list', type=str, default='')
    parser.add_argument('--use_default_list', action='store_true')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

if __name__ == '__main__':
    '''
    cmd1: python model_combine/task1_model_combine.py --use_default_list --select_cnt 5 --output_file attention_cat_0.4_512_combine.pred.json --combine_file_list attention_k_fold_list_1
    cmd2: python model_combine/task1_model_combine.py --use_default_list --select_cnt 6 --output_file attention_cat_0.5_512_combine.pred.json --combine_file_list attention_k_fold_list_2
    cmd3: python model_combine/task1_model_combine.py --use_default_list --select_cnt 3 --output_file attention_cat_0.4_768_combine.pred.json --combine_file_list attention_k_fold_list_3
    cmd4: python model_combine/task1_model_combine.py --use_default_list --select_cnt 3 --output_file combine_test.ann.json --combine_file_list final_combine_list
    '''
    parser = argparse.ArgumentParser(description="task1 model combine")
    arg = parse_arguments(parser)  
    select_cnt = arg.select_cnt
    output_file = os.path.join(data_root, arg.output_file)
    if arg.use_default_list:
        if arg.combine_file_list not in default_combine_list.keys():
            print('%s not exist' %arg.combine_file_list)
            exit(1)
        input_file_list = [os.path.join(data_root, file) for file in default_combine_list[arg.combine_file_list]]
    else:
        combine_file_list = [file.strip() for file in arg.combine_file_list.split(',')]
        input_file_list = [os.path.join(data_root, file) for file in combine_file_list]
    combine(input_file_list, output_file, select_cnt, item_freq_dict_output_file)
