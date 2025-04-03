# -*- coding:utf-8 -*-
import os
# 指定物理GPU 1,2,3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print("[INFO] Set CUDA_VISIBLE_DEVICES to 1,2,3 (avoid GPU0).")

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import numpy as np
import argparse
import pathlib
from tqdm import tqdm
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--noise_type', type=str, default='rand1', help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100')
parser.add_argument('--noise_path', type=str, default=None, help='path of CIFAR-10_human.pt')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar100')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs for training each fold')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--output_dir_prob', type=str, default='./cifar-10-100n/cross_validated_predicted_probabilities', help='directory to save predicted probabilities')
parser.add_argument('--output_dir_label', type=str, default='./cifar-10-100n/cross_validated_predicted_labels', help='directory to save predicted labels')
def main(args):

    def adjust_learning_rate(optimizer, epoch):
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 把noise_type转换
    noise_type_map = {
        'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 
        'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 
        'clean100': 'clean_label', 'noisy100': 'noisy_label'
    }
    args.noise_type = noise_type_map[args.noise_type]

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 确保输出目录存在
    os.makedirs(args.output_dir_prob, exist_ok=True)
    os.makedirs(args.output_dir_label, exist_ok=True)

    # 用于存储最终预测概率
    output_file = os.path.join(
        args.output_dir_prob,
        f"{args.dataset}_trainset_pyx_{args.noise_type}_{args.k_folds}fold_{args.n_epoch}epoch.npy"
    )
    output_file_pred_label = os.path.join(
        args.output_dir_label,
        f"{args.dataset}_trainset_pyx_argmax_predicted_labels_{args.noise_type}_{args.k_folds}fold_{args.n_epoch}epoch.npy"
    )

    # 加载数据集
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './cifar-10-100n/data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './cifar-10-100n/data/CIFAR-100_human.pt'
        else:
            raise NameError(f'Undefined dataset {args.dataset}')

    train_dataset, _, num_classes, num_training_samples = input_dataset(
        args.dataset, args.noise_type, args.noise_path, is_human=False
    )

    # 初始化存储预测概率的数组
    pred_probs = np.zeros((num_training_samples, num_classes), dtype=np.float64)
    pred_labels = np.zeros((num_training_samples,), dtype=np.float64)

    # K折交叉验证
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # 依次进行每个fold的训练和验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(num_training_samples))):
        print(f"Fold {fold + 1}/{args.k_folds}")

        # 划分训练集和验证集
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=args.num_workers)

        # --------------------- 关键修改：多GPU并行 ---------------------
        # 1. 先在单卡上实例化模型
        single_model = ResNet34(num_classes).cuda()
        # 2. 用 DataParallel 包裹单卡模型, device_ids=[0,1,2] 表示使用当前可见的三张卡
        model = nn.DataParallel(single_model, device_ids=[0, 1, 2])
        # -----------------------------------------------------------

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

        # 训练模型
        for epoch in range(args.n_epoch):
            model.train()
            print(f"[Fold {fold + 1}] Training epoch {epoch+1}/{args.n_epoch}")
            for images, labels, _ in tqdm(train_loader, desc=f"Epoch = {epoch + 1}", leave=False):
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
            adjust_learning_rate(optimizer, epoch)    

        # 验证集
        model.eval()
        print(f"[Fold {fold + 1}] Validating ...")
        with torch.no_grad():
            val_data_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
            for images, _, indexes in tqdm(val_data_loader, desc=f"Fold = {fold + 1}", leave=False):
                images = images.cuda()
                logits = model(images)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()
                pred_probs[indexes.numpy()] = probabilities
                pred_labels[indexes.numpy()] = np.max(probabilities, axis=1)


    # 保存预测概率为 .npy 文件
    pred_probs = pred_probs.astype(np.float16)  # 转为 float16 减少文件大小
    # 重新获取每个样本的 argmax
    pred_labels = pred_probs.argmax(axis=1).astype(np.uint16)

    np.save(output_file, pred_probs)
    np.save(output_file_pred_label, pred_labels)
    print(f"Predicted probabilities saved to {output_file}")
    print(f"Predicted argmax_labels saved to {output_file_pred_label}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
