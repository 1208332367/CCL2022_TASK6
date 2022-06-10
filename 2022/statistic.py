import os
import json
import numpy
with open("./ECOM2022/data/train.doc.json") as f:
    train_list = json.load(f)

example_statistic = []
lengths = []
max_lens = []
sentence_max = 0
for example in train_list:
    Descriptor = example['Descriptor']
    event_id = Descriptor['event_id']
    text = Descriptor['text']
    Doc = example['Doc']
    content = Doc['content']
    max_lens.append(len(content))
    sentences = ''
    for sentence in content:
        sentences += sentence['sent_text']
        sentence_max = max(sentence_max, len(sentence['sent_text']))
    lengths.append(len(sentences))
    # example_statistic.append({'doc_id': Doc['doc_id'],
    #                 'length': len(sentences)})

max_lens = numpy.array(max_lens)
print(max_lens.max())
print(max_lens.argmax())
# print(example_statistic)
print(sentence_max)
lengths_array = numpy.array(lengths)
print(lengths_array.max())
print(lengths_array.argmax())
print(train_list[lengths_array.argmax()])
example = train_list[lengths_array.argmax()]
Doc = example['Doc']
content = Doc['content']
print(len(content))
str = ''
for sentence in content:
    str += sentence['sent_text']
# print(len(str))
# split_list = [str[i] for i in range(len(str))]
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# tokenized_list = tokenizer.tokenize(str)
# print(split_list)
# print(tokenized_list)
#
# print(len(tokenized_list))

