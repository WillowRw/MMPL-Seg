import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import WeightedDiceBCE, WeightedDiceCE

class ClipLoss(nn.Module):
    def __init__(self):
        super(ClipLoss, self).__init__()

    def forward(self, logits_per_image, logits_per_text):
        # Validate floating-point tensor inputs if stricter checks are required.
        # if not (torch.is_tensor(logits_per_image) and torch.is_tensor(
        #         logits_per_text) and logits_per_image.type() == 'torch.FloatTensor' and logits_per_text.type() == 'torch.FloatTensor'):
        #     raise ValueError("Input matrices must be floating-point tensors.")

        if logits_per_image.size() != logits_per_text.size():
            raise ValueError("Input matrices must have the same size.")

        labels = torch.arange(
            logits_per_image.shape[0],
            device=logits_per_image.device,
            dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, logits_per_image, logits_per_text):
        if logits_per_image.size() != logits_per_text.size():
            raise ValueError("Input matrices must have the same size.")
        elementwise_difference = (logits_per_image - logits_per_text).pow(2)
        loss_function = nn.SmoothL1Loss(reduction='mean')
        result = loss_function(elementwise_difference, torch.zeros_like(elementwise_difference))
        return result


class LossPoly(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd  # Similarity-loss coefficient.
        # self.seg_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

    def similarity_criterion(self, matrix1, matrix2):
        elementwise_difference = (matrix1 - matrix2).pow(2)
        loss_function = nn.SmoothL1Loss(reduction='mean')
        result = loss_function(elementwise_difference, torch.zeros_like(elementwise_difference))
        return result

    def forward(self, out_image, segment_image, image_similarity, text_similarity):
        # The similarity term operates directly on the two feature matrices.
        seg_loss = self.seg_criterion(out_image, segment_image)
        similarity_loss = self.similarity_criterion(image_similarity, text_similarity)
        final_loss = seg_loss + self.lambd * similarity_loss
        # print("seg_loss ", seg_loss.item(), "similarity_loss", similarity_loss.item(), "final_loss", final_loss.item())
        return final_loss


if __name__ == '__main__':
    B, num_cls, H, W = 8, 2, 256, 256
    out_image = torch.randn(B, num_cls, H, W)
    segment_image = torch.randn(B, num_cls, H, W)
    image_similarity = torch.randn(B, B)
    text_similarity = torch.randn(B, B)
    criterion = LossPoly(0.5)
    loss = criterion(out_image, segment_image, image_similarity, text_similarity)
    print(loss)
