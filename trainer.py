# -*- coding: utf-8 -*-
# @Time : 2024/5/17 14:45
# @Author : Liang yu Gong
# @FileName: trainer.py
# @Software: PyCharm

import os
import sys
import torch
import argparse
import random
import numpy as np

from torch.utils.data import DataLoader
from dataset import Deepfake_Dataset
from data_aug import get_aug
from backbones.model import swin_tiny_patch4_window7_224 as create_model
from loss import *
from adaptive_loss import AdaptiveLossWeight
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc
from loguru import logger


def get_free_gpu():
    """
    自动选择空闲显存最多的GPU
    返回: 设备字符串，如 'cuda:7'
    """
    if not torch.cuda.is_available():
        logger.warning("没有检测到CUDA，使用CPU")
        return 'cpu'

    try:
        # 获取可用GPU数量
        num_gpus = torch.cuda.device_count()
        logger.info(f"检测到 {num_gpus} 个GPU")

        if num_gpus == 0:
            logger.warning("没有检测到GPU，使用CPU")
            return 'cpu'

        if num_gpus == 1:
            return 'cuda:0'

        # 获取每个GPU的显存信息
        max_free_memory = 0
        selected_gpu = 0

        for i in range(num_gpus):
            try:
                # 获取GPU属性
                props = torch.cuda.get_device_properties(i)
                # 获取已分配显存
                allocated = torch.cuda.memory_allocated(i)
                # 获取预留显存
                reserved = torch.cuda.memory_reserved(i)
                # 计算空闲显存（总显存 - 已分配）
                total_memory = props.total_memory
                free_memory = total_memory - allocated

                # 转换为GB显示
                total_gb = total_memory / (1024**3)
                free_gb = free_memory / (1024**3)
                allocated_gb = allocated / (1024**3)

                logger.info(f"GPU {i}: {props.name} | 总显存: {total_gb:.2f}GB | "
                           f"已用: {allocated_gb:.2f}GB | 空闲: {free_gb:.2f}GB")

                # 选择空闲显存最多的GPU
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    selected_gpu = i

            except Exception as e:
                logger.warning(f"获取GPU {i} 信息时出错: {e}")
                continue

        logger.info(f"自动选择GPU {selected_gpu} (空闲显存最多: {max_free_memory/(1024**3):.2f}GB)")
        return f'cuda:{selected_gpu}'

    except Exception as e:
        logger.error(f"选择GPU时出错: {e}，默认使用cuda:0")
        return 'cuda:0'


# 配置 loguru 日志
logger.add("training.log", rotation="10 MB", retention="10 days", level="INFO")
logger.add("training_error.log", rotation="10 MB", retention="10 days", level="ERROR")

if __name__ == '__main__':

    # ADD THE ARGS REQUIRED
    parser = argparse.ArgumentParser(description="Deepfake Swin-Fake Training in Pytorch")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--weights', type=str, default='./checkpoints/swin_tiny_patch4_window7_224_22k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument("--consistency_rate", type=float, default=0.5,
                        help='固定权重时使用 (0-1)，使用自适应权重时此参数无效')
    parser.add_argument('--adaptive-weight', type=bool, default=True,
                        help='是否使用自适应损失权重 (默认True)')
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'], help='Device to use for training (auto/cuda/cpu)')
    parser.add_argument('--gpu-id', type=int, default=7,
                        help='指定GPU编号 (0-7), 默认自动选择')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting Deepfake Detection Training")
    logger.info(f"Arguments: {args}")
    logger.info("=" * 60)

    # Auto-detect device
    if args.device == 'auto':
        if args.gpu_id is not None:
            # 如果指定了GPU ID，使用指定的
            device = torch.device(f'cuda:{args.gpu_id}')
            logger.info(f"使用指定的 GPU: cuda:{args.gpu_id}")
        else:
            # 自动选择空闲显存最多的GPU
            selected_device = get_free_gpu()
            device = torch.device(selected_device)
    elif args.device == 'cuda':
        if args.gpu_id is not None:
            device = torch.device(f'cuda:{args.gpu_id}')
            logger.info(f"使用指定的 GPU: cuda:{args.gpu_id}")
        else:
            device = torch.device('cuda')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        logger.info(f"Loaded pretrain model: {model.load_state_dict(weights_dict, strict=False)}")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                logger.info(f"Training layer: {name}")

    # Define optimizer
    pg = [p for p in model.parameters() if p.requires_grad]

    # 如果使用自适应权重，添加权重参数到优化器
    if args.adaptive_weight:
        loss_weight = AdaptiveLossWeight(num_losses=2).to(device)
        pg.extend(loss_weight.parameters())
        logger.info(f"使用自适应损失权重，初始 log_var: {loss_weight.log_var.data}")
    else:
        loss_weight = None
        logger.info(f"使用固定权重: Consistency={args.consistency_rate}, CE={1-args.consistency_rate}")

    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    # Data augmentations
    aug_list = ['base', 'RE', 'DFDC_Selium', 'RA', 'RandCrop']
    trans_name = random.sample(aug_list, 1)[0]
    transforms = get_aug(name=trans_name, img_size=224)
    logger.info(f"使用数据增强: {trans_name}")

    # Define Datasets
    train_files = os.path.join(os.getcwd(), 'txt_files', 'new_train.txt')
    val_files = os.path.join(os.getcwd(), 'txt_files', 'new_test.txt')

    trans_train_set = Deepfake_Dataset(train_files, transform=transforms)
    train_Loader = DataLoader(dataset=trans_train_set, batch_size=args.batch, shuffle=True)

    trans_val_set = Deepfake_Dataset(val_files, transform=transforms)
    val_Loader = DataLoader(dataset=trans_val_set, batch_size=args.batch, shuffle=True)

    logger.info(f"训练集大小: {len(trans_train_set)}, 验证集大小: {len(trans_val_set)}")
    logger.info(f"Batch size: {args.batch}, Epochs: {args.epochs}")

    best_acc = 0.0
    best_auc = 0.0

    # Model Training:
    for epoch in range(args.epochs):
        model.train()
        loss_function = torch.nn.CrossEntropyLoss()
        accu_loss = torch.zeros(1).to(device)
        accu_num = torch.zeros(1).to(device)

        # 收集训练数据用于计算 AUC
        all_train_labels = []
        all_train_probs = []

        optimizer.zero_grad()
        sample_num = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Training")
        logger.info(f"{'='*60}")

        for step, data in enumerate(train_Loader):
            video_list, labels = data
            batch_size = len(video_list)  # 实际batch中的视频数量（最后一批可能不足batch_size）
            sample_num += batch_size  # 统计视频数量，不是帧数量

            # 视频标签（每个视频的标签，取第一帧即可，因为8帧标签相同）
            video_labels = labels[:, 0].long().to(device)  # [batch_size]

            feature_list = [model(video.to(device)) for video in video_list]

            # 计算所有视频的 Consistency Loss
            L_con_list = []
            for features in feature_list:
                L_con_list.append(Cosine_similarity(features[0]))
            L_con = torch.stack(L_con_list).mean().to(device)

            # 计算所有视频的分类结果
            pred = torch.stack([features[1] for features in feature_list])  # [batch, 8, 2]

            # 视频级预测：对8帧的预测概率取平均
            # pred: [batch, 8, 2] -> 平均 -> [batch, 2]
            pred_probs = torch.softmax(pred, dim=2)  # [batch, 8, 2]
            video_pred_probs = pred_probs.mean(dim=1)  # [batch, 2]
            video_pred_classes = torch.argmax(video_pred_probs, dim=1)  # [batch]

            # 用于AUC计算（使用视频级平均概率）
            video_probs_for_auc = video_pred_probs[:, 1].detach().cpu().numpy()
            all_train_probs.extend(video_probs_for_auc)
            all_train_labels.extend(video_labels.cpu().numpy())

            # 视频级准确率：预测类别 vs 视频标签
            accu_num += torch.eq(video_pred_classes, video_labels).sum()

            # Loss仍然用帧级计算（为了更好的梯度）
            pred_flat = torch.flatten(pred, 0, 1)  # [batch*8, 2]
            labels_flat = torch.flatten(labels, 0, 1).long().to(device)  # [batch*8]
            ce_loss = loss_function(pred_flat, labels_flat)

            # 使用自适应权重或固定权重
            if args.adaptive_weight and loss_weight is not None:
                total_loss, weights = loss_weight([L_con, ce_loss])
                weight_con, weight_ce = weights[0], weights[1]
            else:
                total_loss = args.consistency_rate * L_con + (1 - args.consistency_rate) * ce_loss
                weight_con, weight_ce = args.consistency_rate, 1 - args.consistency_rate

            total_loss.backward()
            accu_loss += total_loss.detach()

            optimizer.step()
            optimizer.zero_grad()

            # 每 10 步记录一次
            if (step + 1) % 10 == 0:
                current_loss = accu_loss.item() / (step + 1)
                current_acc = accu_num.item() / sample_num
                logger.info(f"[Epoch {epoch+1} Step {step+1}/{len(train_Loader)}] "
                           f"Loss: {current_loss:.4f}, Video Acc: {current_acc:.4f}, "  # 明确标注Video Acc
                           f"L_con: {L_con.item():.4f}, L_ce: {ce_loss.item():.4f}")

        # 计算训练集 AUC
        train_auc = roc_auc_score(all_train_labels, all_train_probs) if len(set(all_train_labels)) > 1 else 0.0
        train_loss = accu_loss.item() / len(train_Loader)
        train_acc = accu_num.item() / sample_num

        logger.info(f"\n[Epoch {epoch+1} Training Summary]")
        logger.info(f"  Loss: {train_loss:.4f}")
        logger.info(f"  Video Accuracy: {train_acc:.4f}")  # 明确标注Video Accuracy
        logger.info(f"  Video AUC: {train_auc:.4f}")  # 明确是视频级AUC
        if args.adaptive_weight:
            logger.info(f"  Weights - Consistency: {weight_con:.4f}, CE: {weight_ce:.4f}")

        # Validating
        model.eval()
        with torch.no_grad():
            accu_num = torch.zeros(1).to(device)
            accu_loss = torch.zeros(1).to(device)
            sample_num = 0

            # 收集验证数据用于计算 AUC
            all_val_labels = []
            all_val_probs = []
            all_val_preds = []

            logger.info(f"\n[Epoch {epoch+1} Validation]")

            for step, data in enumerate(val_Loader):
                video_list, labels = data
                batch_size = len(video_list)
                sample_num += batch_size  # 统计视频数量

                # 视频标签
                video_labels = labels[:, 0].long().to(device)  # [batch_size]

                feature_list = [model(video.to(device)) for video in video_list]

                pred = torch.stack([features[1] for features in feature_list])  # [batch, 8, 2]

                # 视频级预测：平均8帧的概率
                pred_probs = torch.softmax(pred, dim=2)  # [batch, 8, 2]
                video_pred_probs = pred_probs.mean(dim=1)  # [batch, 2]
                video_pred_classes = torch.argmax(video_pred_probs, dim=1)  # [batch]

                # 收集视频级预测用于AUC
                all_val_probs.extend(video_pred_probs[:, 1].cpu().numpy())
                all_val_labels.extend(video_labels.cpu().numpy())
                all_val_preds.extend(video_pred_classes.cpu().numpy())

                # 视频级准确率
                accu_num += torch.eq(video_pred_classes, video_labels).sum()

                # Loss仍然用帧级计算
                pred_flat = torch.flatten(pred, 0, 1)
                label_flat = torch.flatten(labels, 0, 1).long().to(device)
                loss = loss_function(pred_flat, label_flat)
                accu_loss += loss

                # 每 10 步记录一次
                if (step + 1) % 10 == 0:
                    current_loss = accu_loss.item() / (step + 1)
                    current_acc = accu_num.item() / sample_num
                    logger.info(f"  [Step {step+1}/{len(val_Loader)}] "
                               f"Loss: {current_loss:.4f}, Video Acc: {current_acc:.4f}")  # 明确标注Video Acc

            # 计算验证集指标
            val_loss = accu_loss.item() / len(val_Loader)
            val_acc = accu_num.item() / sample_num
            val_auc = roc_auc_score(all_val_labels, all_val_probs) if len(set(all_val_labels)) > 1 else 0.0

            # 计算 ROC 曲线数据
            fpr, tpr, thresholds = roc_curve(all_val_labels, all_val_probs)

            logger.info(f"\n[Epoch {epoch+1} Validation Summary]")
            logger.info(f"  Loss: {val_loss:.4f}")
            logger.info(f"  Video Accuracy: {val_acc:.4f}")  # 明确标注Video Accuracy
            logger.info(f"  Video AUC: {val_auc:.4f}")  # 明确是视频级AUC
            logger.info(f"  Best threshold: {thresholds[np.argmax(tpr - fpr)]:.4f}")
            logger.info(f"  TPR: {tpr[np.argmax(tpr - fpr)]:.4f}, FPR: {fpr[np.argmax(tpr - fpr)]:.4f}")

            # 保存最佳模型
            if val_acc > best_acc or val_auc > best_auc:
                logger.info(f"\n{'='*60}")
                logger.info(f"Saving model at Epoch {epoch+1}")

                if val_acc > best_acc:
                    logger.info(f"Best accuracy improved: {best_acc:.4f} -> {val_acc:.4f}")
                    best_acc = val_acc

                if val_auc > best_auc:
                    logger.info(f"Best AUC improved: {best_auc:.4f} -> {val_auc:.4f}")
                    best_auc = val_auc

                # 保存模型
                if args.adaptive_weight and loss_weight is not None:
                    torch.save({
                        'model': model.state_dict(),
                        'loss_weight': loss_weight.state_dict(),
                        'epoch': epoch,
                        'accuracy': val_acc,
                        'auc': val_auc,
                        'args': args
                    }, './checkpoints/Swin_face_v2.pth')
                    logger.info(f"自适应权重 - Consistency: {weight_con:.4f}, CE: {weight_ce:.4f}")
                else:
                    torch.save({
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'accuracy': val_acc,
                        'auc': val_auc,
                        'args': args
                    }, './checkpoints/Swin_face_v2.pth')
                logger.info(f"{'='*60}\n")

    logger.info(f"\n{'='*60}")
    logger.info("Training Completed!")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"{'='*60}")