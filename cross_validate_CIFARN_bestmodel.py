# -*- coding:utf-8 -*-
import os
# 指定物理GPU 1,2,3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print("[INFO] Set CUDA_VISIBLE_DEVICES to 1,2,3 (avoid GPU0).")

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import numpy as np
import argparse
import pathlib
from tqdm import tqdm  # 导入 tqdm 用于进度条

# 命令行参数ss
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--noise_type', type=str, default='worst', help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100')
parser.add_argument('--noise_path', type=str, default=None, help='path of CIFAR-10_human.pt')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar100')
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs for training each fold') # 原来n_epoch = 100
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--output_dir', type=str, default='./cifar-10-100n/cross_validated_predicted_probabilities', help='directory to save predicted probabilities')
parser.add_argument('--output_dir_labels', type=str, default='./cifar-10-100n/cross_validated_predicted_labels', help='directory to save predicted labels')
start_epoch = 0
def adjust_learning_rate(optimizer, epoch):
    # 添加学习率调整计划 把学习率调整为与MIXUP的学习率一样
    alpha_plan = [0.1] * 100 + [0.01] * 50 + [0.001] * 50
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]

def evaluate(val_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.cuda()
            labels = labels.cuda()
            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
    acc = 100 * float(correct) / float(total)
    return acc

def main(args):
    
    # 把noise_type转换
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    args.noise_type = noise_type_map[args.noise_type]

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_labels, exist_ok=True)
    #用于存储交叉预测概率
    output_file = os.path.join(
        args.output_dir,
        f"Best_model_{args.dataset}_trainset_pyx_{args.noise_type}_{args.k_folds}fold.npy"
    )
    
    #用于存储预测标签（每个样本中，仅保留预测概率最大的那一个）
    output_file_pred_label = os.path.join(
        args.output_dir_labels,
        f"Best_model_{args.dataset}_trainset_pyx_argmax_predicted_labels_{args.noise_type}_{args.k_folds}fold.npy"
    )
    
    # 加载数据集
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './cifar-10-100n/data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './cifar-10-100n/data/CIFAR-100_human.pt'
        else:
            raise NameError(f'Undefined dataset {args.dataset}')

    # 只加载训练集，用于做K折交叉验证，获得shape = (50000, 10) 的预测概率文件
    train_dataset, _, num_classes, num_training_samples = input_dataset(
        args.dataset, args.noise_type, args.noise_path, is_human=False
    )

    # 初始化存储预测概率的数组
    pred_probs = np.zeros((num_training_samples, num_classes), dtype=np.float64) # 存各个class的预测概率
    pred_labels = np.zeros((num_training_samples,), dtype=np.float64) # 存最大预测概率

    # K 折交叉验证
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed) #


    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(range(num_training_samples)), desc="Folds", total=args.k_folds)):
        
        use_cuda = torch.cuda.is_available()
        # 初始化模型、优化器
        if use_cuda:
            
            #model = ResNet34(num_classes).cuda()

            # --------------------- 关键修改：多GPU并行 ---------------------
            # 1. 先在单卡上实例化模型
            single_model = ResNet34(num_classes).cuda()
            # 2. 用 DataParallel 包裹单卡模型, device_ids=[0,1,2] 表示使用当前可见的三张卡
            model = nn.DataParallel(single_model)
            ng = torch.cuda.device_count()
            print(f"[INFO] DataParallel mode - using {ng} GPUs.")
            
            # 打印脚本当前可见的每张 GPU 的名称
            print("Devices in use:")
            for i in range(ng):
                device_name = torch.cuda.get_device_name(i)
                print(f"  - Local index {i}: {device_name}")
            print(torch.cuda.device_count())#打印设备号
            cudnn.benchmark = True #benchmark模式，自动寻找当前配置最高效的算法
            print('Using CUDA..')                
            # -----------------------------------------------------------   
        else:
            print("No cuda available!")

        #把优化器实例化
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

        #划分fold - k 的训练集和验证集
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=args.num_workers)                                
        
        # 在每个fold的训练中
        best_acc = 0
        best_model_state = None        

        for epoch in range(start_epoch, args.n_epoch):   
            # 训练模型
            model.train() # 这句话好像可以不写在这里
            adjust_learning_rate(optimizer, epoch)
            for images, labels, _ in tqdm(train_loader, desc=f"Fold = {fold+1}/{args.k_folds}, Epoch = {epoch + 1}/{args.n_epoch}", leave=False) :
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad() # 清除优化器中的所有参数
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward() # 反向传播
                optimizer.step() # 更新参数
            #验证
            val_acc = evaluate(val_loader, model)
            print(f'Validation accuracy at epoch {epoch+1}: {val_acc:.2f}%')
            
            #检查当前Epoch在验证集上的准确率是不是达到了历史最佳，若更好，保存模型状态
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.state_dict().copy() # 这里应该被更换为best_model_state = copy.deepcopy(model.state_dict())，导致后续训练时，模型参数被修改
        
        # 使用最佳模型进行预测
        model.load_state_dict(best_model_state)
        model.eval()
        print(f"Validation Fold {fold + 1}/{args.k_folds}")
        with torch.no_grad():
            for images, _, indexes in tqdm(val_loader,desc=f"Epoch = {epoch + 1}",leave=False):
                images = images.cuda()
                logits = model(images)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()
                pred_probs[indexes.numpy()] = probabilities  # 保存验证集的预测概率
                # pred_labels[indexes.numpy()] = np.max(probabilities, axis=1)  # np.max返回最大值，np.argmax返回最大值的索引
                

    # 保存预测概率为 .npy 文件
    pred_probs = pred_probs.astype(np.float16)  # 先用float32
    # 获取每个样本的argmax
    pred_labels = pred_probs.argmax(axis=1).astype(np.uint16) # 预测标签是0-9的整数，不用float32

    # 将文件保存到对应的文件夹
    np.save(output_file, pred_probs)
    print(f"Predicted probabilities saved to {output_file}")
    np.save(output_file_pred_label, pred_labels)
    print(f"Predicted argmax_labels saved to {output_file_pred_label}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)