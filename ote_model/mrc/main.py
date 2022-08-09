from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup as WarmupLinearSchedule

from utils_squad import (combine_train_dev, save_train_dev, read_squad_examples, convert_examples_to_features, 
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

# k折交叉验证
def get_kfold_data(k, i, X):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid = X[val_start:val_end]
        X_train = X[0:val_start] + X[val_end:]
    else:  # 若是最后一折交叉验证
        X_valid = X[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]

    return X_train, X_valid

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_precision = 0
    results = dict()
    results['precision'] = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      'token_type_ids':  None if args.model_type == 'xlm' else batch[2],
                      'start_positions': batch[3],
                      'end_positions':   batch[4]}
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask':    batch[6]})
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if results['precision'] > best_precision:
                    best_precision = results['precision']
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)  # 0713 add
                    logger.info("Saving best model to %s", output_dir)
                     
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    logger.info("Best performance: " + str(best_precision)) # 0713 add

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    #print('---eval_dataloader start---')
    #print(len(dataset))
    #print('---eval_dataloader end---')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.predict_file,
                        model.config.start_n_top, model.config.end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)

    def get_clear_text(text):
        text = text.strip().split()
        res = ''.join(text)
        if '”' in res and '“' not in res:
            res = '“' + res
        if '“' in res and '”' not in res:
            res = res + '”'
        return res.replace(',', ' ').strip()

    corr_num = 0
    for example in examples:
        idx = example.qas_id
        pred_aspect = all_predictions[idx]
        gold_aspect = example.orig_answer_text
        if get_clear_text(pred_aspect) == gold_aspect: 
            corr_num += 1
    result = dict()
    if len(examples):
        result['precision'] = corr_num / len(examples)
    else:
        result['precision'] = 0

    output_file = os.path.join(args.result_dir, args.result_file.format(prefix))
    opinions = []
    for example in examples:
        if args.language == 'english':
            opinion = {'event_id': example.event_id, 'doc_id': example.doc_id,
                        'start_sent_idx': example.start_sent_idx, 'end_sent_idx': example.end_sent_idx,
                        'argument': all_predictions[example.qas_id]}
        else:   
            opinion = {'event_id': example.event_id, 'doc_id': example.doc_id,
                        'start_sent_idx': example.start_sent_idx, 'end_sent_idx': example.end_sent_idx,
                        'argument': get_clear_text(all_predictions[example.qas_id])}
        opinions.append(opinion)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(opinions, ensure_ascii=False))
        logger.info('Writing result to: ' + output_file)
    if len(examples):
        logger.info('Eval Precision: ' + str(corr_num / len(examples)) + '\t' + str(corr_num) + '\t' + str(len(examples)))

    return result


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = args.data_dir + args.predict_file if evaluate else args.data_dir + args.train_file
    
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                       language=args.language,
                                       opinion_level=args.opinion_level,
                                       word_seg_model=args.word_seg_model)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)
    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--k_fold', type=int, default=-1, help="k_fold")
    parser.add_argument('--word_seg_model', type=str, default='jieba', help="Word segment model")
    parser.add_argument("--train_file", default='train', type=str,
                        help="Training file.")
    parser.add_argument("--predict_file", default='dev', type=str,
                        help="Predicting file.")
    parser.add_argument("--test_file", default='test', type=str,
                        help="Testing file.")
    parser.add_argument("--data_dir", default='../../data/', type=str)
    parser.add_argument("--result_dir", default='../../result/chinese_result', type=str)
    parser.add_argument("--result_file", default='mrc.ann.json', type=str)
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default='../../model_files/bert_sqad/', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument('--opinion_level', type=str, default='segment', choices=['segment', 'sent'])

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=384, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--language", type=str, default='english')

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--device', type=str, default="cpu",
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'],
                        help="GPU/CPU devices")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    device = args.device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.k_fold != -1:
            k_fold = args.k_fold
            logger.info(" start %d fold cross valid", k_fold)
            src_train_file = args.train_file
            src_pred_file = args.predict_file
            src_output_dir = args.output_dir
            src_result_file = args.result_file
            train_file = args.data_dir + src_train_file
            dev_file = args.data_dir + src_pred_file
            dataset = combine_train_dev(train_file, dev_file)
            dataset = dataset[:len(dataset) // k_fold * k_fold]
            for i in range(k_fold):
                logger.info(" start k = %d", i)
                tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
                model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
                if args.local_rank == 0:
                    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
                model.to(args.device)
                args.result_file = src_result_file
                train_data, dev_data = get_kfold_data(k_fold, i, dataset)
                args.train_file = f'k_fold_{i}_' + src_train_file
                args.predict_file = f'k_fold_{i}_' + src_pred_file
                args.output_dir = src_output_dir.strip('/') + f'_k_fold_{i}/'
                train_file = args.data_dir + args.train_file
                dev_file = args.data_dir + args.predict_file
                save_train_dev(train_file, dev_file, train_data, dev_data)
                train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
                global_step, tr_loss = train(args, train_dataset, model, tokenizer)
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
                logger.info(" finish fold k = %d", i)
                args.result_file = f'k_fold_{i}_' + src_result_file
                args.predict_file = args.test_file
                do_test(args, model_class, tokenizer_class)
        else:
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            args.predict_file = args.test_file
            do_test(args, model_class, tokenizer_class)

        exit(0)

    args.predict_file = args.test_file
    if args.k_fold != -1:
        k_fold = args.k_fold
        src_result_file = args.result_file
        src_output_dir = args.output_dir
        for i in range(k_fold):
            args.result_file = f'k_fold_{i}_' + src_result_file
            args.output_dir = src_output_dir.strip('/') + f'_k_fold_{i}/'
            do_test(args, model_class, tokenizer_class)
    else:
        do_test(args, model_class, tokenizer_class)

def do_test(args, model_class, tokenizer_class):
    # Save the trained model and the tokenizer
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        ''' 
        if args.do_train:
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(args.output_dir)
        '''
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
    

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    """
    train_cmd1(mrc_jieba_train_accum3_epoch6):nohup python ote_model/mrc/main.py --word_seg_model jieba --num_train_epochs 6 --overwrite_cache --do_train  --train_file train  --predict_file dev --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch6/  --data_dir data/ECOB-ZH/   --result_dir result/chinese_result --device cuda:0 > logs/mrc_jieba_train_accum3_epoch6.train.log 2>&1 &
    train_cmd2(mrc_ltp_train_accum3_epoch5):nohup python ote_model/mrc/main.py --word_seg_model ltp --num_train_epochs 5 --overwrite_cache --do_train  --train_file train  --predict_file dev --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5   --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_ltp_train_accum3_epoch5/  --data_dir data/ECOB-ZH/   --result_dir result/chinese_result --device cuda:0 > logs/mrc_ltp_train_accum3_epoch5.train.log 2>&1 &
    train_cmd3(mrc_ltp_enhance_train_accum3_epoch6):nohup python ote_model/mrc/main.py --word_seg_model ltp --num_train_epochs 6 --overwrite_cache --do_train  --train_file enhance_train  --predict_file dev --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5   --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_ltp_enhance_train_accum3_epoch6/  --data_dir data/ECOB-ZH/   --result_dir result/chinese_result --device cuda:0 > logs/mrc_ltp_enhance_train_accum3_epoch6.train.log 2>&1 &
    train_cmd4(mrc_jieba_train_accum3_epoch5_cross_valid):nohup python ote_model/mrc/main.py --k_fold 10 --word_seg_model jieba --num_train_epochs 5 --overwrite_cache --do_train  --train_file train  --predict_file dev --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch5_cross_valid/  --data_dir data/ECOB-ZH/   --result_dir result/chinese_result --device cuda:0 > logs/mrc_jieba_train_accum3_epoch5_cross_valid.train.log 2>&1 &
    
    test_cmd1(mrc_jieba_train_accum3_epoch6):nohup python ote_model/mrc/main.py --result_file mrc_jieba_train_accum3_epoch6_best.ann.json --word_seg_model jieba --num_train_epochs 6 --overwrite_cache  --train_file train  --predict_file test --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch6/  --data_dir data/ECOB-ZH/   --result_dir model_combine/task2_combine_data --device cuda:0 > logs/mrc_jieba_train_accum3_epoch6.test.log 2>&1 &
    test_cmd2(mrc_ltp_train_accum3_epoch5):nohup python ote_model/mrc/main.py --result_file mrc_ltp_train_accum3_epoch5_best.ann.json --word_seg_model ltp --num_train_epochs 5 --overwrite_cache --train_file train  --predict_file test --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5   --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_ltp_train_accum3_epoch5/  --data_dir data/ECOB-ZH/   --result_dir model_combine/task2_combine_data --device cuda:0 > logs/mrc_ltp_train_accum3_epoch5.test.log 2>&1 &
    test_cmd3(mrc_ltp_enhance_train_accum3_epoch6):nohup python ote_model/mrc/main.py --result_file mrc_ltp_enhance_train_accum3_epoch6_best.ann.json --word_seg_model ltp --num_train_epochs 6 --overwrite_cache  --train_file enhance_train  --predict_file test --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5   --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_ltp_enhance_train_accum3_epoch6/  --data_dir data/ECOB-ZH/   --result_dir model_combine/task2_combine_data --device cuda:0 > logs/mrc_ltp_enhance_train_accum3_epoch6.test.log 2>&1 &
    test_cmd4(mrc_jieba_train_accum3_epoch5_cross_valid):nohup python ote_model/mrc/main.py --k_fold 10 --word_seg_model jieba --num_train_epochs 5 --overwrite_cache  --train_file train  --predict_file test --gradient_accumulation_steps 3  --language chinese  --model_name_or_path pretrained_model/chinese_pretrain_mrc_roberta_wwm_ext_large  --do_eval --do_lower_case --learning_rate 3e-5 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=2  --evaluate_during_training --output_dir model_files/chinese_model/mrc_jieba_train_accum3_epoch5_cross_valid/  --data_dir data/ECOB-ZH/   --result_dir model_combine/task2_combine_data --device cuda:0 > logs/mrc_jieba_train_accum3_epoch5_cross_valid.test.log 2>&1 &
    """
    main()