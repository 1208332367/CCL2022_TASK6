import torch
import torch.nn as nn

from crf import LinearCRF
from encoder import Encoder


class NNCRF(nn.Module):

    def __init__(self, config):
        super(NNCRF, self).__init__()
        self.device = config.device
        self.encoder = Encoder(config)
        self.inferencer = LinearCRF(config)

    def forward(self,
                sent_seq_lens: torch.Tensor,
                sent_tensor: torch.Tensor,
                event_tensor: torch.Tensor,
                tags: torch.Tensor
                ):
        _, scores = self.encoder(sent_tensor, event_tensor)
        batch_size = sent_tensor.size(0)
        sent_len = sent_tensor.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(
            self.device)
        mask = torch.le(maskTemp, sent_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        # Inference.
        unlabed_score, labeled_score = self.inferencer(scores, sent_seq_lens, tags, mask)
        return unlabed_score - labeled_score

    def decode(self, batchInput):
        """
        Decode the batch input
        """
        sent_seq_lens, sent_tensor, event_tensor, label_seq_tensor = batchInput
        # Encode.
        feature_out, features = self.encoder(sent_tensor, event_tensor)
        # Decode.
        bestScores, decodeIdx = self.inferencer.decode(features, sent_seq_lens)
        return bestScores, decodeIdx
