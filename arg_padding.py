import json
import argparse

def loadJson(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def argPadding(file_in, file_out):
    items = loadJson(file_in)
    print(len(items))
    for item in items:
        item['argument'] = ""
    with open(file_out, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=4)

def parse_arguments(parser):
    # parameters
    parser.add_argument('--input_file', type=str, default='result/chinese_result/seq.pred.json')
    parser.add_argument('--output_file', type=str, default='data/ECOB-ZH/test.ann.json')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

if __name__ == '__main__':
    '''
    cmd: python arg_padding.py --input_file model_combine/task1_combine_data/combine_test.ann.json --output_file data/ECOB-ZH/test.ann.json
    '''
    parser = argparse.ArgumentParser(description="argument padding")
    arg = parse_arguments(parser)  
    argPadding(arg.input_file, arg.output_file)