import functools

import mmcv
import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.models import weighted_loss
from mmdet.models.builder import LOSSES
from torch import nn


@mmcv.jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(
        loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        raise ValueError('avg_factor must be provided')
    if reduction == 'mean':
        return loss.sum() / avg_factor
    if reduction != 'none':
        raise ValueError('avg_factor cannot be used with reduction="sum"')
    return loss


def custom_weighted_dir_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        return custom_weight_dir_reduce_loss(
            loss, weight, reduction, avg_factor)

    return wrapper


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    return torch.abs(pred - target)


@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    num_samples, num_dir, _ = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    target_sign = target.new_ones((num_samples, num_dir)).flatten(0)
    loss = loss_func(
        pred.flatten(0, 1), target.flatten(0, 1), target_sign)
    return loss.view(num_samples, num_dir)


@LOSSES.register_module()
class PtsDirCosLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override or self.reduction
        return self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction,
            avg_factor=avg_factor)


@LOSSES.register_module()
class PtsL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override or self.reduction
        return self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction,
            avg_factor=avg_factor)


@MATCH_COST.register_module()
class OrderedPtsL1Cost:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        num_gts, num_orders, _, _ = gt_bboxes.shape
        bbox_pred = bbox_pred.reshape(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.flatten(2).reshape(num_gts * num_orders, -1)
        return torch.cdist(bbox_pred, gt_bboxes, p=1) * self.weight
