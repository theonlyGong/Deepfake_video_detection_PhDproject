# -*- coding: utf-8 -*-
# @Time : 2024/4/4 16:25
# @Author : Liangyu Gong
# @FileName: loss.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


def Cosine_similarity(feature_list):
    """
    计算特征列表中所有特征对的余弦相似度损失
    使用 PyTorch 原生函数，避免 CPU/GPU 来回切换
    """
    if len(feature_list) < 2:
        return torch.tensor(0.0, device=feature_list[0].device)

    sim_list = []

    for feat_1, feat_2 in combinations(feature_list, 2):
        # 使用 PyTorch 原生 cosine_similarity，保持在 GPU 上计算
        cos_sim = F.cosine_similarity(feat_1, feat_2, dim=1)
        # 计算损失：1 - 余弦相似度（让特征更分散）
        sim_list.append(1 - cos_sim.mean())

    loss = torch.stack(sim_list).mean()
    return loss

if __name__ == '__main__':
    a = torch.rand(49,768)
    b = torch.rand(49,768)
    c = torch.rand(49,768)
    fea_list = [a,b,c]
    loss = Cosine_similarity(fea_list)
    print(loss)