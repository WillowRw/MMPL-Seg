import os
from PIL import Image
import numpy as np
import random
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_auc_score, jaccard_score
import torch
from torch.nn.functional import one_hot
from torch import nn
import torch.nn.functional as F
# from skimage.morphology import dilation
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.reshape(-1)
        truth = truth_pixel.view(-1)
        assert (logit.shape == truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-5
        neg_weight = neg.sum().item() + 1e-5
        loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (logit.shape == truth.shape)
        p = logit.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        w = truth.detach()
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        p = w * (p)
        t = w * (t)
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - (2 * intersection + smooth) / (union + smooth)

        loss = dice.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat + targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coefficient
        return dice_loss

class WeightedDiceCE(nn.Module):
    def __init__(self, dice_weight=0.5, CE_weight=0.5):
        super(WeightedDiceCE, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(2)  # OSIC: 4, RITE: 5
        self.CE_weight = CE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        # inputs = inputs.argmax(dim=1)
        dice, dice1, dice2, dice3 = self.dice_loss(inputs, targets)
        hard_dice_coeff = 1 - dice
        dice01 = 1 - dice1
        dice02 = 1 - dice2
        dice03 = 1 - dice3
        torch.cuda.empty_cache()
        return hard_dice_coeff, dice01, dice02, dice03

    def forward(self, inputs, targets):
        targets = targets.long()
        dice_CE_loss = self.dice_loss(inputs, targets)
        torch.cuda.empty_cache()
        return dice_CE_loss


class WeightedDiceBCE_unsup(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE_unsup, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets, LV_loss):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE + 0.1 * LV_loss

        return dice_BCE_loss


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape)
        # print(inputs)
        # print(targets)

        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss


def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    aucs = []
    for i in range(pred.shape[1]):
        prediction = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask = masks[i].cpu().detach().numpy()
        # print("rrr",np.max(mask), np.min(mask))
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)


def iou_on_batch(masks, pred):
    # print(masks.shape, pred.shape)  # torch.Size([8, 2, 224, 224]) torch.Size([8, 2, 224, 224])
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    ious = []
    for i in range(pred.shape[0]):
        pred_tmp = pred[i].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    # print(ious)
    return np.mean(ious)


# def dice_coef(y_true, y_pred):
#     smooth = 1e-5
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef(target, predicted):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice_coefficient = (2.0 * intersection + 1e-5) / (union + 1e-5)

    return dice_coefficient

def dice_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    dices = []
    # print(masks.shape)

    for i in range(pred.shape[0]):
        pred_tmp = pred[i].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp = pred[i]
        mask_tmp = masks[i]
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        dice = dice_coef(mask_tmp, pred_tmp)
        dice = dice.cpu().detach().numpy()
        dices.append(dice)

    return np.mean(dices)


def draw_seg_result(i, sample_img, sample_mask, sample_out, model_name):
    # fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    # fig.suptitle(f'Model: {model_name}', fontsize=16)
    # axes[0].imshow(np.transpose(sample_img, (1, 2, 0)))  # Convert from CHW to HWC.
    # axes[0].set_title('CT Image')
    # axes[0].axis('off')
    # axes[1].imshow(np.squeeze(sample_mask, axis=0), cmap='gray')
    # axes[1].set_title('GT')
    # axes[1].axis('off')
    # axes[2].imshow(sample_out, cmap='gray')
    # axes[2].set_title('Predict')
    # axes[2].axis('off')
    # plt.show()


    plt.figure(figsize=(10, 10))
    # plt.imshow(sample_out, cmap='gray')
    plt.imshow(sample_mask, cmap='gray')
    # plt.imshow(np.transpose(sample_img, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(f"draw/segmentation/{i}.jpg", bbox_inches='tight', pad_inches=0)
    plt.close()


def draw_similarity(image_similarity, text_similarity):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image_similarity, cmap='gray_r', vmin=image_similarity.min(), vmax=image_similarity.max())
    axes[0].set_title('image_similarity')
    axes[0].axis('off')

    # axes[0].annotate(text="", xy=(0, 0), xytext=(0, 0), textcoords="offset points",
    #                  bbox=dict(boxstyle="round", fc="w"))

    axes[1].imshow(text_similarity, cmap='gray_r', vmin=text_similarity.min(), vmax=text_similarity.max())
    axes[1].set_title('text_similarity')
    axes[1].axis('off')

    # axes[1].annotate(text="", xy=(0, 0), xytext=(0, 0), textcoords="offset points",
    #                  bbox=dict(boxstyle="round", fc="w"))

    plt.show()


# Metric helpers.


def dice_coeff(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def accuracy(pred, target):
    correct_pixels = (pred == target).sum().item()
    total_pixels = target.numel()
    return correct_pixels / total_pixels


# Compute metrics for one iteration.
def calculate_metrics_iteration(predicted, target):
    predicted[predicted >= 0.5] = 1
    predicted[predicted < 0.5] = 0
    # target[target > 0] = 1
    # target[target <= 0] = 0
    dice = dice_coeff(predicted, target)
    iou_value = iou(predicted, target)
    acc = accuracy(predicted, target)
    return dice.item(), iou_value.item(), acc


def seed_torch(seed=666):  # 1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def post_processing(mask, kernel_size=3):
    mask = mask.squeeze(0, 1).cpu().numpy()
    # Optional connected-component filtering.
    # num_labels, mask, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # res, mask = cv2.connectedComponents(mask, connectivity=8)  # connectivity=4
    # mask = np.where(mask > 2, 0, mask)

    # Morphological opening and closing.
    mask = np.uint8(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = torch.from_numpy(mask).to(device)
    return mask

# Generate a CutMix bounding box.
def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
