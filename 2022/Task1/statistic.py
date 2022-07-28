import os
import json
import numpy
from matplotlib import pyplot as plt

# with open("./ECOM2022/data/train.doc.json") as f:
#     train_list = json.load(f)
#
# example_statistic = []
# lengths = []
# max_lens = []
# sentence_max = 0
# total = 0
# for example in train_list:
#     Descriptor = example['Descriptor']
#     event_id = Descriptor['event_id']
#     text = Descriptor['text']
#     Doc = example['Doc']
#     content = Doc['content']
#     max_lens.append(len(content))
#     total += len(content)
#     sentences = ''
#     for sentence in content:
#         sentences += sentence['sent_text']
#         sentence_max = max(sentence_max, len(sentence['sent_text']))
#     lengths.append(len(sentences))
#     # example_statistic.append({'doc_id': Doc['doc_id'],
#     #                 'length': len(sentences)})
# print(total)
# max_lens = numpy.array(max_lens)
# print(max_lens.max())
# print(max_lens.argmax())
# # print(example_statistic)
# print(sentence_max)
# lengths_array = numpy.array(lengths)
# print(lengths_array.max())
# print(lengths_array.argmax())
# print(train_list[lengths_array.argmax()])
# example = train_list[lengths_array.argmax()]
# Doc = example['Doc']
# content = Doc['content']
# print(len(content))
# str = ''
# for sentence in content:
#     str += sentence['sent_text']
# # print(len(str))
# # split_list = [str[i] for i in range(len(str))]
# # from transformers import BertTokenizer
# # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# # tokenized_list = tokenizer.tokenize(str)
# # print(split_list)
# # print(tokenized_list)
# #
# # print(len(tokenized_list))
# with open("./ECOM2022/data/enhance_train.ann.json") as f:
#     train_list = json.load(f)
# total = 0
# print(len(train_list))
# for example in train_list:
#     total += example["end_sent_idx"] - example["start_sent_idx"] + 1
# print(total)

with open("./ECOM2022/data/test.doc.json") as f:
    test_doc = json.load(f)
sum = 0
d = []
a = []
for doc in test_doc:
    content = doc["Doc"]["content"]
    for i, sent in enumerate(content):
        if i == len(content) - 1:
            continue
        elif i == len(content) - 2:
            continue
        else:
            d.append(len(content[i]["sent_text"]) + len(content[i + 1]["sent_text"]) + len(content[i + 2]["sent_text"]))
        a.append(len(content[i]["sent_text"]))
# print(d[3666: 3680])
# print(a[3666: 3680])
length = []
for i in range(1000):
    length.append(0)
# print(length)
for i, l in enumerate(d):
    if l > 256:
        sum += 1
    length[l] += 1
x = [i for i in range(1000)]
print(sum)
plt.plot(x, length)
plt.show()
# print(length)
# with open("./baseline_mac.pred.json") as f:
#     pred = json.load(f)
# sum = 0
# for p in pred:
#     sum += 1
# print(sum)
#
# with open("./ECOM2022/data/dev.ann.json") as f:
#     train_ann = json.load(f)
# sum = 0
# for ann in train_ann:
#     if ann["start_sent_idx"] == ann["end_sent_idx"]:
#         sum += 1
# print(sum)
# print(len(train_ann))
