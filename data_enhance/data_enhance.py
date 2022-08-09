import json
import os
import time
import random
import argparse
from tqdm import tqdm

from API_baidu_translate import baiduTranslateAPI
from crawling_youdao_translate import yuodaoTranslateCrawling

data_root = '../data/ECOB-ZH'

train_doc_file = os.path.join(data_root, 'train.doc.json')
dev_doc_file = os.path.join(data_root, 'dev.doc.json')
test_doc_file = os.path.join(data_root, 'test.doc.json')

train_ann_file = os.path.join(data_root, 'train.ann.json')
dev_ann_file = os.path.join(data_root, 'dev.ann.json')

input_file = os.path.join(data_root, 'dev_zh_to_en.doc.json')
output_file = os.path.join(data_root, 'train_zh_to_en.doc.json')

# 载入json文件
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# 存储json文件
def store_json(filename, json_data, json_indent=1, mode='w'):
    with open(filename, mode, encoding='utf-8') as f:
        json.dump(json_data, f, indent=json_indent, ensure_ascii=False)

# 存储json文件
def store_json_with_fp(fp, json_data, json_indent=1):
    json.dump(json_data, fp, indent=json_indent, ensure_ascii=False)

def generate_new_doc_item(src_doc, translate_inst, src_lang, dst_lang):
    contents = src_doc['Doc']['content']
    new_contents = []
    crawling_error = True
    for content in contents:
        q = content['sent_text']
        # print(q)
        if 'trans_code' in content.keys():
            if content['trans_code'] != 0:
                new_contents.append(content)
                continue
        code, res_data = translate_inst.translate(q, src_lang, dst_lang)
        content['trans_code'] = code
        if code != 0:
            print('text [', q, '] translate failed')
            print('errmsg: ', res_data)
            new_contents.append(content)
            continue 
        crawling_error = False  
        content['sent_text'] = res_data
        new_contents.append(content)
        time.sleep(0.1)
    src_doc['Doc']['content'] = new_contents
    return crawling_error, src_doc

def generate_new_doc_file(src_doc_file, dst_doc_file, src_lang, dst_lang, translate_inst, overwrite=True, start_doc_id=0):
    docs = load_json(src_doc_file)

    if overwrite:
        if os.path.exists(dst_doc_file):
            print('file exist, do overwrite')
            os.remove(dst_doc_file)
    with open(dst_doc_file, 'a+', encoding='utf-8') as f:
        if overwrite:
            f.write('[\n')
        idx = 0
        for doc in tqdm(docs):
            if doc['Doc']['doc_id'] < start_doc_id:
                idx += 1
                continue
            crawling_error, new_doc = generate_new_doc_item(doc, translate_inst, src_lang, dst_lang)
            if crawling_error:
                print('-' * 30)
                print('detect crawling error, program exit')
                break
            store_json_with_fp(f, new_doc, json_indent=4)
            if idx < len(docs) - 1:
                f.write(',\n')
            else:
                f.write('\n')
            if idx % 25 == 0:
                time.sleep(random.randint(2, 5))
            idx += 1

        if idx == len(docs):
            f.write(']')

def check_doc_item(src_doc, baidu_translate, src_lang, dst_lang):
    contents = src_doc['Doc']['content']
    new_contents = []
    doc_err_cnt = 0
    for content in contents:       
        if content['trans_code'] == 0:
            new_contents.append(content)
            continue
        doc_err_cnt += 1
        q = content['sent_text']
        print(src_doc['Doc']['doc_id'], q)
        code, res_data = baidu_translate.translate(q, src_lang, dst_lang)
        content['trans_code'] = code
        if code != 0:
            print('text [', q, '] translate failed')
            print('errmsg: ', res_data)
            new_contents.append(content)
            continue 
        content['sent_text'] = res_data
        new_contents.append(content)
        time.sleep(0.1)
    src_doc['Doc']['content'] = new_contents
    return doc_err_cnt, src_doc

def check_doc_file(src_doc_file, dst_doc_file, src_lang, dst_lang, translate_inst):
    src_docs = load_json(src_doc_file)
    dst_docs = []
    total_err_cnt = 0
    for src_doc in tqdm(src_docs):
        doc_err_cnt, dst_doc = check_doc_item(src_doc, translate_inst, src_lang, dst_lang)
        total_err_cnt += doc_err_cnt
        dst_docs.append(dst_doc)
    print('-' * 30)
    print('detect total error count: %d' %total_err_cnt)
    store_json(dst_doc_file, dst_docs, json_indent=4)

def get_last_doc_id(file):
    doc_id_list = []
    with open(file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith('"doc_id":'):
                doc_id_list.append(int(line.strip(',').replace('"doc_id": ', '')))
        print('get last doc_id: %d' %doc_id_list[-1])
        return doc_id_list[-1]

def concat_doc_file(file_list, output_file, doc_id_increase_list):
    if len(file_list) != len(doc_id_increase_list):
        print('length of arg[doc_id_increase_list] should be equal to length of arg[concat_file_list]')
        return -1
    res = []
    for i in range(len(file_list)):
        file = file_list[i]
        doc_id_increase = doc_id_increase_list[i]
        json_file = load_json(os.path.join(data_root, file))
        for item in json_file:
            if 'ann' in file:
                item['doc_id'] += doc_id_increase
                if item not in res:
                    res.append(item)
            else:
                item['Doc']['doc_id'] += doc_id_increase
                doc = item
                sent_list = []
                for sent in item['Doc']['content']:
                    if 'trans_code' in sent.keys():
                        if sent['trans_code'] != 0:
                            print('detect error code in file %s, doc_id: %d' %(file, item['Doc']['doc_id']))
                            return -1
                    sent_list.append({'sent_idx': sent['sent_idx'], 'sent_text': sent['sent_text']})
                doc['Doc']['content'] = sent_list
                res.append(doc)
    store_json(output_file, res, json_indent=4)
    return 0

def parse_arguments(parser):
    # parameters
    parser.add_argument('--check_doc', type=bool, default=False)
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--do_translate', action='store_true')
    parser.add_argument('--do_concat', action='store_true')
    parser.add_argument('--concat_file_list', type=str, default='')
    parser.add_argument('--doc_id_increase_list', type=str, default='')
    parser.add_argument('--src_lang', type=str, default='zh')
    parser.add_argument('--dst_lang', type=str, default='en')
    parser.add_argument('--translate', type=str, default='youdao')
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--random_min', type=int, default=2)
    parser.add_argument('--random_max', type=int, default=5)
    parser.add_argument('--sleep_weight', type=float, default=0.1)

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data enhance")
    arg = parse_arguments(parser)  
    output_file = os.path.join(data_root, arg.output_file)
     
    '''
    cmd1: python data_enhance.py --do_translate --translate baidu --input_file train.doc.json --output_file baidu_train_zh_to_en.doc.json --src_lang zh --dst_lang en
    cmd2: python data_enhance.py --do_translate --translate baidu --input_file check2_train_zh_to_en.doc.json --output_file baidu_train_en_to_zh.doc.json --src_lang en --dst_lang zh
    cmd3(check not translated items): python data_enhance.py --do_translate --check_doc True --translate baidu --input_file check1_baidu_train_en_to_zh.doc.json --output_file check2_baidu_train_en_to_zh.doc.json --src_lang en --dst_lang zh
    cmd4(concat files): python data_enhance.py --do_concat --concat_file_list enhance_train.doc.json,enhance_dev.doc.json --doc_id_increase_list 0,0 --output_file enhance_train_dev.doc.json
    '''
    if arg.do_translate:
        check_doc = arg.check_doc
        input_file = os.path.join(data_root, arg.input_file)
        overwrite = arg.overwrite
        src_lang = arg.src_lang
        dst_lang = arg.dst_lang
        random_min = arg.random_min
        random_max = arg.random_max
        sleep_weight = arg.sleep_weight
        translate_inst = yuodaoTranslateCrawling()
        if arg.translate == 'baidu':
            translate_inst = baiduTranslateAPI()
        if not check_doc:
            start_doc_id = 0
            if not overwrite:
                start_doc_id = get_last_doc_id(output_file) + 1
            generate_new_doc_file(input_file, output_file, src_lang, dst_lang, translate_inst, overwrite=overwrite, start_doc_id=start_doc_id)
        else:
            check_doc_file(input_file, output_file, src_lang, dst_lang, translate_inst)

    if arg.do_concat:
        concat_file_list = [file.strip() for file in arg.concat_file_list.split(',')]   
        doc_id_increase_list = [int(doc_id_increase.strip()) for doc_id_increase in arg.doc_id_increase_list.split(',')]   
        concat_doc_file(concat_file_list, output_file, doc_id_increase_list)