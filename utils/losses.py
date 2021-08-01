import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.nn


# def dice_loss(score, target):
#     target = target.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target * target)
#     z_sum = torch.sum(score * score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     loss = 1 - loss
#     return loss
#
# def dice_loss1(score, target):
#     target = target.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target)
#     z_sum = torch.sum(score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     loss = 1 - loss
#     return loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        targets = targets.float()
        smooth = 1e-5
        inputs = F.softmax(inputs, dim=1)
        # targets= F.softmax(targets, dim=1)
        intersect = torch.sum(inputs * targets)
        y_sum = torch.sum(targets)
        z_sum = torch.sum(inputs)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss


class WeightedCELoss(nn.Module):
    def __init__(self, weight):
        super(WeightedCELoss, self).__init__()
        self.weights = torch.tensor(weight).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')

    def forward(self, inputs, target):
        target = torch.argmax(target, dim=1)
        return self.ce_loss(input=inputs, target=target)


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):

        logit = F.softmax(input, dim=1)
        target = torch.argmax(target, dim=1)
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class focal_loss(nn.Module):
    def __init__(self, pos_weight, gamma=2, num_classes=4, size_average=True, epslion=1e-6):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.eps = epslion
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, self.eps, 1. - self.eps)
        # self.alpha = self.alpha.to(inputs.device)
        # targets = one_hot(targets, num_class=4)
        # targets_pos = targets.sum(dim=(2, 3)) + self.eps
        cross_ent_pos = torch.log(inputs) * targets * torch.pow((1. - inputs), self.gamma)
        cross_ent_pos = cross_ent_pos.sum(dim=(2, 3))
        cross_ent_pos = cross_ent_pos / (inputs.shape[2] * inputs.shape[3])
        return -torch.mean(self.pos_weight * cross_ent_pos)


class CombinedLoss(nn.Module):
    def __init__(self, weights, lambda1=0.5):
        super(CombinedLoss, self).__init__()
        self.weight = weights
        self.ce_loss = WeightedCELoss(weight=weights)
        # self.focal_loss=MultiFocalLoss(num_class=4,alpha=weights,smooth=0.01)
        self.dice_loss = DiceLoss()
        self.lambda1 = lambda1

    def forward(self, input, target):
        ce_loss = self.ce_loss(input, target)
        dice_loss = self.dice_loss(input, target)
        return self.lambda1 * dice_loss + ce_loss, ce_loss, dice_loss


class WeightRegressionLoss(nn.Module):
    def __init__(self, weight):
        super(WeightRegressionLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        assert input.size() == target.size()
        input_softmax = F.softmax(input, dim=1)
        target_softmax = F.softmax(target, dim=1)
        num = input.size(1)
        loss = 0
        for i in range(num):
            loss += self.weight[i] * torch.mean((input_softmax[:, i, :, :] - target_softmax[:, i, :, :]) ** 2)
        return loss / num


class UncertaintyDiceLoss(nn.Module):
    def __init__(self):
        super(UncertaintyDiceLoss, self).__init__()

    def forward(self, inputs, targets, uncertainty):
        targets = targets.float()
        batch_size = inputs.size(0)
        smooth = 1e-5
        inputs = F.softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        total_loss = 0
        for i in range(inputs.size(1)):
            inputs[:, i, :, :] *= uncertainty.squeeze(1)
            intersect = torch.sum(inputs[:, i] * targets[:, i])
            y_sum = torch.sum(targets[:, i])
            z_sum = torch.sum(inputs[:, i])
            loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            total_loss = total_loss + (1 - loss)
        return total_loss / batch_size


class CombinedUncertaintyLoss(nn.Module):
    def __init__(self, probability_weight, contour_weights, distance_weights, lambda1=0.5, lambdal2=0.5):
        super(CombinedUncertaintyLoss, self).__init__()
        self.contour_ce_loss = WeightedCELoss(weight=contour_weights)
        self.probability_ce_loss = WeightedCELoss(weight=probability_weight)
        # self.dice_loss = UncertaintyDiceLoss()
        self.dice_loss = DiceLoss()
        self.reg_loss = WeightRegressionLoss(distance_weights)
        self.lambda1 = lambda1
        self.lambda2 = lambdal2

    def forward(self, probability, distance, contour, target_probability, target_distance, target_contour, uncertainty):
        return self.dice_loss(probability, target_probability) + self.probability_ce_loss(probability,
                                                                                          target_probability) + \
               self.lambda1 * self.contour_ce_loss(contour, target_contour) \
               + self.lambda2 * self.reg_loss(distance, target_distance)



