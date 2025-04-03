# -*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
print("[INFO] Set CUDA_VISIBLE_DEVICES to 1,2,3 (avoid GPU0).")

import copy
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
parser.add_argument('--noise_type', type=str, default='worst', help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100')
parser.add_argument('--noise_path', type=str, default=None, help='path of CIFAR-10_human.pt')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar100')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs for training each fold')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--output_dir_prob', type=str, default='./cifar-10-100n/cross_validated_predicted_probabilities', help='directory to save predicted probabilities')
parser.add_argument('--output_dir_label', type=str, default='./cifar-10-100n/cross_validated_predicted_labels', help='directory to save predicted labels')
parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait before early stop if no progress')

def main(args):
    def adjust_learning_rate(optimizer, epoch):
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    noise_type_map = {
        'clean': 'clean_label', 
        'worst': 'worse_label', 
        'aggre': 'aggre_label',
        'rand1': 'random_label1', 
        'rand2': 'random_label2', 
        'rand3': 'random_label3',
        'clean100': 'clean_label', 
        'noisy100': 'noisy_label'
    }
    args.noise_type = noise_type_map[args.noise_type]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir_prob, exist_ok=True)
    os.makedirs(args.output_dir_label, exist_ok=True)

    output_file = os.path.join(
        args.output_dir_prob,
        f"Best_model_Earlystop_{args.dataset}_{args.noise_type}_{args.k_folds}fold_{args.n_epoch}epoch.npy"
    )
    output_file_pred_label = os.path.join(
        args.output_dir_label,
        f"Best_model_Earlystop_{args.dataset}_predicted_labels_{args.noise_type}_{args.k_folds}fold_{args.n_epoch}epoch.npy"
    )

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

    pred_probs = np.zeros((num_training_samples, num_classes), dtype=np.float64)
    pred_labels = np.zeros((num_training_samples,), dtype=np.float64)

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # 新增：记录每个fold停止的epoch
    fold_stop_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(num_training_samples))):
        print(f"\n========== Fold {fold + 1}/{args.k_folds} ==========")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=args.num_workers)

        single_model = ResNet18(num_classes).cuda()
        model = nn.DataParallel(single_model)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

        best_model_wts = None
        best_acc = 0.0
        best_epoch = 0
        no_improve_counter = 0
        stop_training = False
        final_epoch = 0  # 新增：记录该fold最终停止的epoch

        for epoch in range(args.n_epoch):
            if stop_training:
                break

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

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.cuda(), labels.cuda()
                    logits = model(images)
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_acc = correct / total

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                print(f"No improvement for {no_improve_counter}/{args.patience} epochs")
                if no_improve_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                    stop_training = True
            
            final_epoch = epoch + 1  # 更新最终停止的epoch

        # 记录该fold的停止epoch
        fold_stop_epochs.append(final_epoch)
        
        print(f"[Fold {fold + 1}] Best val acc: {best_acc:.4f} at epoch {best_epoch+1}")
        print(f"[Fold {fold + 1}] Training stopped at epoch {final_epoch}")

        model.load_state_dict(best_model_wts)

        model.eval()
        with torch.no_grad():
            val_data_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=args.num_workers)
            for images, _, indexes in tqdm(val_data_loader, desc=f"Final Prediction (Fold = {fold + 1})", leave=False):
                images = images.cuda()
                logits = model(images)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()
                pred_probs[indexes.numpy()] = probabilities

    # 保存预测概率
    pred_probs = pred_probs.astype(np.float16)
    pred_labels = pred_probs.argmax(axis=1).astype(np.uint16)

    np.save(output_file, pred_probs)
    np.save(output_file_pred_label, pred_labels)
    print(f"\n[INFO] Predicted probabilities saved to {output_file}")
    print(f"[INFO] Predicted argmax_labels saved to {output_file_pred_label}")

    # 新增：打印所有fold的停止epoch
    print("\n[SUMMARY] Training stop epochs for each fold:")
    for fold, stop_epoch in enumerate(fold_stop_epochs, 1):
        print(f"Fold {fold}: Stopped at epoch {stop_epoch}")
    # 计算平均停止epoch
    avg_stop_epoch = sum(fold_stop_epochs) / len(fold_stop_epochs)
    print(f"Average stopping epoch across {args.k_folds} folds: {avg_stop_epoch:.1f}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)