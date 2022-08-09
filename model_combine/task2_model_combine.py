import os
import json
import argparse

data_root = 'model_combine/task2_combine_data'

default_combine_list = {
    'final_combine_list': [
        'k_fold_0_mrc.ann.json',
        'k_fold_1_mrc.ann.json',
        'k_fold_2_mrc.ann.json',
        'k_fold_3_mrc.ann.json',
        'k_fold_4_mrc.ann.json',
        'k_fold_5_mrc.ann.json',
        'k_fold_6_mrc.ann.json',
        'k_fold_7_mrc.ann.json',
        'k_fold_8_mrc.ann.json',
        'k_fold_9_mrc.ann.json',
        'mrc_jieba_train_accum3_epoch6_best.ann.json',
        'mrc_ltp_train_accum3_epoch5_best.ann.json',
        'mrc_ltp_enhance_train_accum3_epoch6_best.ann.json'
    ]
}

item_freq_dict = {}
item_freq_dict_output_file = os.path.join(data_root, 'item_freq_dict.json')

# 载入json文件
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# 存储json文件
def store_json(filename, json_data, json_indent=1):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=json_indent, ensure_ascii=False)

# 计数+1
def increase_item_cnt(item_str, argument):
    if item_str in item_freq_dict.keys():
        if argument in item_freq_dict[item_str].keys():
            item_freq_dict[item_str][argument] += 1
        else:
            item_freq_dict[item_str][argument] = 1
    else:
        item_freq_dict[item_str] = {
            argument: 1
        }

# 标注item转字符串
def item2str(item):
    return '##'.join((str(item['event_id']), str(item['doc_id']), str(item['start_sent_idx']), str(item['end_sent_idx'])))

# 字符串转标注item
def str2item(item_str, argument):
    res_list = item_str.split('##')
    return {
        "event_id": int(res_list[0]),
        "doc_id": int(res_list[1]),
        "start_sent_idx": int(res_list[2]),
        "end_sent_idx": int(res_list[3]),
        "argument": argument
    }

# 标注文件转字符串集合并计数
def item_list_to_str_list(item_list):
    res = []
    for item in item_list:
        item_str = item2str(item) 
        argument = item['argument']
        increase_item_cnt(item_str, argument) # 计数+1
        res.append(item_str)
    return res

# 字符串集合转标注文件
def str_set_to_item_list(str_list):
    res = []
    for item_str in str_list:
        res.append(str2item(item_str))
    return res

# 标注文件融合
def combine(input_file_list, output_file, item_freq_dict_output_file):
    # 生成item频率
    for input_file in input_file_list:
        str_list = item_list_to_str_list(load_json(input_file)) 
        print('input file [%s] items: %d' %(input_file, len(str_list)))
    # 筛选高频item
    store_json(item_freq_dict_output_file, item_freq_dict, json_indent=4)
    combine_list = []
    for item_str, arg_freq in item_freq_dict.items():
        max_cnt = 0
        best_argument = ''
        for argument, cnt in arg_freq.items():
            if cnt > max_cnt:
                max_cnt = cnt
                best_argument = argument
        combine_list.append(str2item(item_str, best_argument))
    # print(item_freq_dict)
    print('-' * 30)
    print('combine file [%s] items: %d' %(output_file, len(combine_list)))
    store_json(output_file, combine_list, json_indent=4)

def parse_arguments(parser):
    # parameters
    parser.add_argument('--output_file', type=str, default='combine_mrc.ann.json')
    parser.add_argument('--combine_file_list', type=str, default='')
    parser.add_argument('--use_default_list', action='store_true')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

if __name__ == '__main__':
    '''
    cmd: python model_combine/task2_model_combine.py --use_default_list --output_file team_g9OrQLtH.ann.json --combine_file_list final_combine_list
    '''
    parser = argparse.ArgumentParser(description="task2 model combine")
    arg = parse_arguments(parser)  
    output_file = os.path.join(data_root, arg.output_file)
    if arg.use_default_list:
        if arg.combine_file_list not in default_combine_list.keys():
            print('%s not exist' %arg.combine_file_list)
            exit(1)
        input_file_list = [os.path.join(data_root, file) for file in default_combine_list[arg.combine_file_list]]
    else:
        combine_file_list = [file.strip() for file in arg.combine_file_list.split(',')]
        input_file_list = [os.path.join(data_root, file) for file in combine_file_list]
    combine(input_file_list, output_file, item_freq_dict_output_file)
