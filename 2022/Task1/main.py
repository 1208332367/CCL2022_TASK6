import argparse
import os
import pickle
import random
import time
from typing import List
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
from transformers import BertTokenizer
from NNCRF import NNCRF
from config import Config
from eval import evaluate_batch_insts
from func import set_seed, read_data, batching_list_instances, simple_batching, get_optimizer, write_results, lr_decay
from instance import Instance


def parse_arguments(parser):
    # Training hyper parameters
    parser.add_argument('--device', type=str, default="cpu",
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--data_dir', type=str, default="./ECOM2022/data/")
    parser.add_argument('--result_dir', type=str, default="./result/chinese_result/")
    parser.add_argument('--model_dir', type=str, default="./model_files/")
    parser.add_argument('--model_folder', type=str, default="model", help="The name to save the model files")
    parser.add_argument('--train_file', type=str, default="train")
    parser.add_argument('--dev_file', type=str, default="dev")
    parser.add_argument('--test_file', type=str, default="test")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1, help="default batch size is 1")
    parser.add_argument('--accumulation_steps', type=int, default=2, help="default accumulation_steps is 2")
    parser.add_argument('--num_epochs', type=int, default=6, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=5,
                        help="early stop when there is n epoch not increasing on dev")

    # model hyperparameter
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--sentence_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256, help="hidden size of the LSTM")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=0, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--bert', type=str, default='bert-base-chinese')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_insts=None, dev_insts=None, test_insts=None):
    model = NNCRF(config)
    model = model.to(config.device)
    tokenizer = BertTokenizer.from_pretrained(config.bert)

    # Count the number of parameters.
    num_param = 0
    for idx in list(model.parameters()):
        try:
            num_param += idx.size()[0] * idx.size()[1]
        except IndexError:
            num_param += idx.size()[0]
    print('num_param: ', num_param)

    # Get optimizer.
    optimizer = get_optimizer(config, model)
    # Get instances.
    train_num = len(train_insts)
    print("number of instances: %d" % train_num)
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config.batch_size, train_insts)
    dev_batches = batching_list_instances(config.batch_size, dev_insts, shffule=False)
    test_batches = batching_list_instances(config.batch_size, test_insts, shffule=False)

    best_dev = [-1, 0, -1]
    best_test = [-1, 0, -1]

    # Path to save op_extract_model.
    model_folder = config.model_folder

    model_dir = config.model_dir + model_folder
    model_path = model_dir + f"/{model_folder}_crf.m"
    config_path = model_dir + f"/{model_folder}_config.conf"
    res_path = config.result_dir + f"{model_folder}.pred.json"

    # Create new dirs.
    print("[Info] The model will be saved to: %s" % model_path)
    print("[Info] The config will be saved to: %s" % config_path)
    print("[Info] The result will be saved to: %s" % res_path)
    os.makedirs(model_dir, exist_ok=True)  # create model files. not raise error if exist.

    # Training model
    no_incre_dev = 0
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        step = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in tqdm(np.random.permutation(len(batched_data))):
            sent_seq_len, sent_tensor, event_tensor, label_seq_tensor = simple_batching(config, batched_data[index], tokenizer)
            model.train()
            loss = model(sent_seq_len, sent_tensor, event_tensor, label_seq_tensor)
            loss = loss / config.accumulation_steps
            epoch_loss += loss.item()
            loss.backward()
            if (step+1) % config.accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
            step += 1
        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)
        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts, tokenizer)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts, tokenizer)
        if dev_metrics[2] > best_dev[0]:
            print(colored("saving the best model...", 'blue'))
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_path)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_path, test_insts)
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev" % no_incre_dev)
            break


def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance], tokenizer):
    print(colored('[Begin evaluating ' + name + '.]', 'green'))
    metrics, metrics_e2e = np.asarray([0, 0, 0], dtype=int), np.zeros((1, 3), dtype=int)
    batch_idx = 0
    batch_size = config.batch_size

    # Calculate metrics by batch.
    with torch.no_grad():
        for batch in tqdm(batch_insts_ids):
            # get instances list.
            one_batch_insts = insts[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            # get ner result.
            processed_batched_data = simple_batching(config, batch, tokenizer)
            batch_max_scores, batch_max_ids = model.decode(processed_batched_data)

            # evaluate ner result.
            # get the num of correctly predicted arguments, predicted arguments and gold arguments.
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, processed_batched_data[-1],
                                            processed_batched_data[0], config.idx2labels)

            batch_idx += 1
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    print("Opinion Extraction: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
    print("Correctly predicted opinion: %d, Total predicted opinion: %d, Total golden opinion: % d" % (p, total_predict, total_entity))
    return [precision, recall, fscore]


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)

    # update
    opt.train_file = opt.data_dir + opt.train_file
    opt.dev_file = opt.data_dir + opt.dev_file
    opt.test_file = opt.data_dir + opt.test_file
    conf = Config(opt)

    set_seed(opt, conf.seed)

    # Read train/test/dev data into instance.
    trains = read_data(conf.train_file, conf.train_num)
    devs = read_data(conf.dev_file, conf.dev_num)
    tests = read_data(conf.test_file, conf.test_num)
    # trains[0].print_info()
    # Data Preprocess.
    # Relabel IBO labels to IOBES labels.
    trains = conf.use_iobes(trains)
    devs = conf.use_iobes(devs)
    tests = conf.use_iobes(tests)
    conf.build_label_idx(trains + devs + tests)

    conf.build_word_idx(trains, devs, tests)

    conf.map_insts_ids(trains)
    conf.map_insts_ids(devs)
    conf.map_insts_ids(tests)
    # trains[0].print_info()
    print("num chars: " + str(conf.num_char))
    print("num words: " + str(len(conf.word2idx)))
    # print(conf.char2idx)
    # Train model.
    train_model(conf, conf.num_epochs, trains, devs, tests)


if __name__ == "__main__":
    """
    command:
    python eoe_model/seq/main.py \
       --lr 5e-4 \
       --backbone_lr 1e-6 \
       --batch_size 2 \
       --bert bert-base-chinese \
       --num_epochs 10 \
       --data_dir data/ECOB-ZH/ \
       --model_dir model_files/chinese_model/ \
       --result_dir result/chinese_result/
    """
    main()
