import torch


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


loss = torch.nn.BCELoss(reduce=False)


def MaskBECLoss(pred, label, valid_len):
    weights = torch.ones_like(label)
    weights = sequence_mask(weights, valid_len)
    unweighted_loss = loss(pred, label)
    weighted_loss = (unweighted_loss * weights).mean()
    return weighted_loss

