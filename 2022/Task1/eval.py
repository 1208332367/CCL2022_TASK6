from instance import Instance
from typing import List
import numpy as np
import torch


class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        """
        self.left = left
        self.right = right

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash((self.left, self.right))


def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str]) -> np.ndarray:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    p = 0
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        # Get gold labels of a passage.
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        # Get predicted labels of a passage.
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        # Covert ids to labels.
        output = [idx2label[l] for l in output]
        prediction = [idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction

        # Get gold arguments.
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("S"):
                output_spans.add(Span(i, i, output[i][2:]))
            if output[i].startswith("B"):
                start = i
            if output[i].startswith("E"):
                end = i
                if start != -1 and start <= end:
                    output_spans.add(Span(start, end, output[i][2:]))
                    start = -1

        # Get predicted span.
        predict_spans = set()
        start = -1
        for i in range(len(prediction)):
            if prediction[i].startswith("S"):
                predict_spans.add(Span(i, i, prediction[i][2:]))
            if prediction[i].startswith("B"):
                start = i
            if prediction[i].startswith("E"):
                end = i
                if start != -1 and start <= end:
                    predict_spans.add(Span(start, end, prediction[i][2:]))
                    start = -1
        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))

    return np.asarray([p, total_predict, total_entity], dtype=int)
