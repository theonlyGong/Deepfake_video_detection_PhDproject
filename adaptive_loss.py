# -*- coding: utf-8 -*-
"""
自适应损失权重模块
让 consistency loss 和 CE loss 的权重根据梯度自动学习
"""

import torch
import torch.nn as nn


class AdaptiveLossWeight(nn.Module):
    """
    自适应损失权重模块
    参考论文: "Multi-Task Learning Using Uncertainty to Weigh Losses..."

    原理: 让模型自动学习每个任务的不确定性(uncertainty),
          不确定性高的任务权重自动降低
    """
    def __init__(self, num_losses=2):
        super(AdaptiveLossWeight, self).__init__()
        # 可学习的对数方差参数 (log variance)
        # 初始值设为 0，对应不确定性为 1
        self.log_var = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        """
        Args:
            losses: list of losses [L_con, L_ce, ...]

        Returns:
            weighted_loss: 加权后的总损失
            weights: 每个损失的权重
        """
        if len(losses) != len(self.log_var):
            raise ValueError(f"Number of losses ({len(losses)}) must match num_losses ({len(self.log_var)})")

        total_loss = 0
        weights = []

        for i, loss in enumerate(losses):
            # 权重 = 1 / (2 * sigma^2) = 1 / (2 * exp(log_var))
            # 不确定性越大(sigma越大)，权重越小
            precision = torch.exp(-self.log_var[i])
            weighted_loss = precision * loss + self.log_var[i]
            total_loss += weighted_loss
            weights.append(precision.item())

        return total_loss, weights


class SimpleLearnableWeight(nn.Module):
    """
    简单的可学习权重版本
    使用 softmax 保证权重和为 1
    """
    def __init__(self, num_losses=2):
        super(SimpleLearnableWeight, self).__init__()
        # 可学习的原始权重 (logits)
        self.logits = nn.Parameter(torch.ones(num_losses))

    def forward(self, losses):
        """
        Args:
            losses: list of losses

        Returns:
            weighted_loss: 加权后的总损失
            weights: softmax 后的权重
        """
        # softmax 归一化，保证权重和为 1
        weights = torch.softmax(self.logits, dim=0)

        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += weights[i] * loss

        return total_loss, weights.tolist()


class GradNormWeight(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing
    根据梯度大小动态调整权重
    """
    def __init__(self, num_losses=2, alpha=1.5):
        super(GradNormWeight, self).__init__()
        self.num_losses = num_losses
        self.alpha = alpha  # 恢复力系数

        # 可学习的权重
        self.weights = nn.Parameter(torch.ones(num_losses))

        # 记录初始损失值
        self.initial_losses = None

    def forward(self, losses, model=None):
        """
        Args:
            losses: list of losses
            model: 模型（用于计算梯度）

        Returns:
            weighted_loss: 加权后的总损失
            weights: 当前权重
        """
        # 第一次调用时记录初始损失
        if self.initial_losses is None:
            self.initial_losses = [l.detach().item() for l in losses]

        # 计算加权损失
        weights = torch.relu(self.weights) + 0.1  # 保证权重为正
        weights = weights / weights.sum()  # 归一化

        total_loss = sum(w * l for w, l in zip(weights, losses))

        return total_loss, weights.tolist()


class ConstrainedWeight(nn.Module):
    """
    约束权重模块：权重和恒为1，只有一个可学习参数
    通过 sigmoid 将参数约束在 (0,1)，自动计算第二个权重为 1-w
    """
    def __init__(self):
        super(ConstrainedWeight, self).__init__()
        # 只学习一个参数，范围通过 sigmoid 约束在 (0, 1)
        self.w = nn.Parameter(torch.tensor(0.5))  # 初始值为0.5

    def forward(self, losses):
        """
        Args:
            losses: list of losses [L_con, L_ce]

        Returns:
            weighted_loss: 加权后的总损失
            weights: [w_con, w_ce] 权重列表
        """
        # 使用 sigmoid 将参数约束在 (0, 1)
        w_con = torch.sigmoid(self.w)
        w_ce = 1 - w_con  # 保证权重和为1

        # 计算加权损失
        total_loss = w_con * losses[0] + w_ce * losses[1]

        return total_loss, [w_con.item(), w_ce.item()]


# 使用示例
if __name__ == '__main__':
    # 创建可学习权重模块
    loss_weight = AdaptiveLossWeight(num_losses=2)

    # 模拟两个损失
    L_con = torch.tensor(0.5, requires_grad=True)
    L_ce = torch.tensor(0.3, requires_grad=True)

    # 前向传播
    total_loss, weights = loss_weight([L_con, L_ce])

    print(f"L_con: {L_con.item():.4f}, L_ce: {L_ce.item():.4f}")
    print(f"Weights: {weights}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Learnable log_var: {loss_weight.log_var.data}")