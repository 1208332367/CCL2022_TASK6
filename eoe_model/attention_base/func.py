import json
import random

import torch
import numpy as np
from termcolor import colored
from torch import optim

from instance import Instance
from typing import List


def read_data(file: str, number: int = 5) -> List[Instance]:
    print("Reading " + file + " file.")
    insts = []
    # Read document.
    with open(file + '.doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)
    # Read annotation.
    try:
        with open(file + '.ann.json', 'r', encoding='utf-8') as f:
            opinions = json.load(f)
    except FileNotFoundError:
        opinions = []
        print(colored('[There is no ' + file + '.ann.json.]', 'red'))
    for doc in docs:
        event = doc['Descriptor']['text']
        event_id = doc['Descriptor']['event_id']
        doc_id = doc['Doc']['doc_id']
        title = doc['Doc']['title']
        contents = doc['Doc']['content']
        sents = [content['sent_text'] for content in contents]
        labels = ['O'] * len(sents)
        targets = ['O'] * len(sents)
        doc_opinions = [opinion for opinion in opinions if int(opinion['doc_id']) == int(doc_id)]
        for opinion in doc_opinions:
            for sent_idx in range(opinion['start_sent_idx'], opinion['end_sent_idx'] + 1):
                labels[sent_idx] = 'I'
                targets[sent_idx] = opinion['argument']
            labels[opinion['start_sent_idx']] = 'B'
        inst = Instance(doc_id, sents, event, event_id, title, labels, targets)
        insts.append(inst)
    if number > 0:
        insts = insts[:number]
    print("Number of documents: {}".format(len(insts)))
    return insts


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], 1, vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def batching_list_instances(batch_size, insts: List[Instance], shffule=True):
    """
    List of instances -> List of batches
    """
    if shffule:
        insts.sort(key=lambda x: len(x.input))
    train_num = len(insts)
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(one_batch_insts)
    if shffule:
        random.shuffle(batched_data)
    return batched_data


def simple_batching(config, insts, tokenizer, word_pad_idx=0):
    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        sent_seq_len: Shape: (batch_size), the length of each paragraph in a batch.
        sent_tensor: Shape: (batch_size, max_seq_len, max_token_num)
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_data = insts
    batch_size = len(batch_data)
    word_pad_idx = word_pad_idx
    # doc len
    doc_sents = [inst.input for inst in insts]  # doc_num * doc_len
    events = [inst.event for inst in insts]
    max_sent_len = max([len(doc_sent) for doc_sent in doc_sents])
    sent_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input), batch_data)))
    event_len = torch.LongTensor(list(map(lambda inst: len(inst.event), batch_data)))
    # sent tensor
    doc_sent_ids = []
    max_token_len = 0
    events_ids = []
    max_event_len = 0
    # descriptor
    for idx, doc in enumerate(doc_sents):
        # 语句tokenizer baseline
        # sent_ids = [tokenizer.encode_plus(sent, events[idx]).input_ids for sent in doc]  # doc_len * token_num
        # 语句tokenizer 仅sentence
        sent_ids = [tokenizer.encode_plus(sent).input_ids for sent in doc]  # doc_len * token_num
        # 语句tokenizer 使用语句的前后语句作为额外补充
        # sent_ids = concat(doc, tokenizer)
        # 事件tokenizer
        # event_ids = tokenizer.encode_plus(events[idx]).input_ids
        event_ids = tokenizer.encode_plus(events[idx], add_special_tokens=False).input_ids
        max_event_len = max(max_event_len, len(event_ids))
        max_token_len = max(max_token_len, max([len(sent) for sent in sent_ids]))
        events_ids.append(event_ids)
        doc_sent_ids.append(sent_ids)
    # padding: batch_size, max_sent_len, max_token_len
    for doc_idx, doc_sent_id in enumerate(doc_sent_ids):
        for sent_idx, sent_id in enumerate(doc_sent_id):
            pad_token_num = - len(sent_id) + max_token_len
            doc_sent_ids[doc_idx][sent_idx].extend([word_pad_idx] * pad_token_num)
        pad_sent_num = max_sent_len - len(doc_sent_id)
        for i in range(pad_sent_num):
            doc_sent_ids[doc_idx].append([word_pad_idx] * max_token_len)
    # padding: batch_size, max_event_len
    for event_idx, event_ids in enumerate(events_ids):
        pad_word_num = max_event_len - len(event_ids)
        events_ids[event_idx].extend([word_pad_idx] * pad_word_num)
    # label seq tensor
    label_seq_tensor = torch.zeros((batch_size, max_sent_len), dtype=torch.long)
    for idx in range(batch_size):
        if batch_data[idx].output_ids:
            label_seq_tensor[idx, :sent_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)

    # list to tensor
    sent_seq_len = sent_seq_len.to(config.device)
    sent_tensor = torch.LongTensor(doc_sent_ids).to(config.device)
    # event_len = event_len.to(config.device)
    event_tensor = torch.LongTensor(events_ids).to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)

    return sent_seq_len, sent_tensor, event_tensor, label_seq_tensor


def concat(doc, tokenizer):
    sent_ids = []
    for i, sent in enumerate(doc):
        sent_id = tokenizer.encode_plus(sent).input_ids
        if i == 0:
            sent_nextid = tokenizer.encode_plus(doc[i + 1], add_special_tokens=False).input_ids
            sent_id.extend(sent_nextid)
            sent_id.append(102)
        elif i == len(doc) - 1:
            sent_preid = tokenizer.encode_plus(doc[i - 1], add_special_tokens=False).input_ids
            sent_id.extend(sent_preid)
            sent_id.append(102)
        else:
            sent_preid = tokenizer.encode_plus(doc[i - 1], add_special_tokens=False).input_ids
            sent_nextid = tokenizer.encode_plus(doc[i + 1], add_special_tokens=False).input_ids
            sent_id.extend(sent_preid)
            sent_id.append(102)
            sent_id.extend(sent_nextid)
            sent_id.append(102)
        # if i == len(doc) - 1:
        #     sent_nextid = tokenizer.encode_plus(doc[i - 1], add_special_tokens=False).input_ids
        #     sent_id.extend(sent_nextid)
        #     sent_id.append(102)
        #     sent_nextid = tokenizer.encode_plus(doc[i - 2], add_special_tokens=False).input_ids
        #     sent_id.extend(sent_nextid)
        #     sent_id.append(102)
        # elif i == len(doc) - 2:
        #     sent_nextid = tokenizer.encode_plus(doc[i + 1], add_special_tokens=False).input_ids
        #     sent_id.extend(sent_nextid)
        #     sent_id.append(102)
        #     sent_nextid = tokenizer.encode_plus(doc[i - 1], add_special_tokens=False).input_ids
        #     sent_id.extend(sent_nextid)
        #     sent_id.append(102)
        # else:
        #     sent_nextid = tokenizer.encode_plus(doc[i + 1], add_special_tokens=False).input_ids
        #     sent_id.extend(sent_nextid)
        #     sent_id.append(102)
        #     sent_nextid = tokenizer.encode_plus(doc[i + 2], add_special_tokens=False).input_ids
        #     sent_id.extend(sent_nextid)
        #     sent_id.append(102)
        sent_id = sent_id[:230]
        sent_ids.append(sent_id)
    return sent_ids


def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def get_optimizer(config, model):
    """
    Method to get optimizer.
    """
    base_params = list(map(id, model.encoder.LM.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": config.lr},
        {"params": model.encoder.LM.parameters(), "lr": config.backbone_lr},
    ]

    if config.optimizer.lower() == "sgd":
        print(
            colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params)
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def use_ibo(label):
    if label.startswith('E'):
        label = 'I'
    elif label.startswith('S'):
        label = 'B'
    return label


def write_results(filename: str, insts):
    """
    Save results.
    Each json instance is an opinion.
    """
    opinions = []
    for inst in insts:
        event_id = inst.event_id
        doc_id = inst.doc_id
        labels = inst.prediction

        start_sent_idx = -1
        for sent_idx, label in enumerate(labels):
            if label == 'S':
                opinion = {'event_id': event_id, 'doc_id': doc_id, 'start_sent_idx': sent_idx,
                           'end_sent_idx': sent_idx}
                opinions.append(opinion)
            if label == 'B':
                start_sent_idx = sent_idx
            if label == 'E':
                end_sent_idx = sent_idx
                if start_sent_idx != -1 and start_sent_idx <= end_sent_idx:
                    opinion = {'event_id': event_id, 'doc_id': doc_id, 'start_sent_idx': start_sent_idx,
                               'end_sent_idx': end_sent_idx}
                    opinions.append(opinion)
                    start_sent_idx = -1
    with open(filename, 'w', encoding='utf-8') as f:
        print("writing", len(opinions), " results to ", filename)
        f.write(json.dumps(opinions))



