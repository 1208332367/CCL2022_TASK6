import os
import random
import numpy as np
import torch
from torch import nn, optim

PAD, START, STOP = "<PAD>", "<START>", "<STOP>"


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


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='none')
NLLLoss = torch.nn.NLLLoss(reduction='none')
softmax = nn.Softmax(dim=-1)


def MaskedSoftmaxCrossEntropyLoss(pred, label, valid_len):
    """带遮蔽的softmax交叉熵损失函数"""
    weights = torch.ones_like(label)
    weights = sequence_mask(weights, valid_len)
    unweighted_loss = CrossEntropyLoss(pred.permute(0, 2, 1), label)
    weighted_loss = (unweighted_loss * weights).sum()
    return weighted_loss


def MaskedSoftmaxNLLLoss(pred, label, valid_len, n):
    """带遮蔽的softmax交叉熵损失函数"""
    weights = torch.ones_like(label)
    weights = sequence_mask(weights, valid_len)
    # print(weights)
    pred = softmax(pred)
    # print(pred)
    pow_weight = torch.ones_like(pred)
    pow_weight[:, :, 1] = n
    pow_weight[:, :, 2] = n
    # print(pow_weight)
    pred = torch.pow(pred, pow_weight)
    pred = torch.log(pred)
    # print(pred)
    # print(label)
    unweighted_loss = NLLLoss(pred.permute(0, 2, 1), label)
    # print(unweighted_loss)
    weighted_loss = (unweighted_loss * weights).sum()
    return weighted_loss


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
