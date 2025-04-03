# -*- coding: utf-8 -*-
import os
import argparse
import torch
from PIL import Image
from data.datasets import input_dataset

def save_cifar_images_to_dir(images, labels, root_dir="train"):
    """
    将 CIFAR 风格的 numpy 数组和标签保存到指定目录下，目录结构示例:
      root_dir/class_0/00000.png
      root_dir/class_0/00001.png
      root_dir/class_1/00000.png
      ...
    
    参数:
    - images: numpy 数组，形状 (N, 32, 32, 3)
    - labels: list 或 numpy 数组，长度 N
    - root_dir: 最终保存图片的根目录
    """

    os.makedirs(root_dir, exist_ok=True)

    for idx, (img_array, label) in enumerate(zip(images, labels)):
        # 将标签转为字符串，用作子文件夹名
        class_str = str(label)
        class_dir = os.path.join(root_dir, class_str)
        os.makedirs(class_dir, exist_ok=True)

        # 从 numpy 数组转换为 PIL Image
        img_pil = Image.fromarray(img_array)
        # 命名并保存文件
        save_path = os.path.join(class_dir, f"{idx:05d}.png")
        img_pil.save(save_path)

    print(f"已将 {len(images)} 张图片保存到 {root_dir}/ 下，按类别分文件夹。")

def main():
    parser = argparse.ArgumentParser(description="将 CIFAR-N 数据集转换为图像文件夹格式")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='选择数据集，可选：cifar10 或 cifar100')
    parser.add_argument('--noise_type', type=str, default='rand1',
                        help='可选：clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100')
    parser.add_argument('--noise_path', type=str, default=None,
                        help='CIFAR-10_human.pt 或 CIFAR-100_human.pt 的路径')
    parser.add_argument('--is_human', action='store_true', default=False,
                        help='是否使用人工标注的噪声标签')
    parser.add_argument('--save_test', action='store_true', default=False,
                        help='是否同时保存测试集图片')
    args = parser.parse_args()

    # 将命令行的 noise_type 转换成真实 .pt 文件内对应的键名
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
    # 若超出上述范围，可能会 KeyError
    if args.noise_type in noise_type_map:
        args.noise_type = noise_type_map[args.noise_type]
    else:
        raise ValueError(f"不支持的 noise_type: {args.noise_type}")

    # 若未指定 noise_path，根据 dataset 设置默认值
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './cifar-10-100n/data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './cifar-10-100n/data/CIFAR-100_human.pt'
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")

    # ============== 加载数据集 ==============
    # 需要确保你的环境里有 data.datasets 模块，且其中包含 input_dataset 函数。
    # 该函数通常返回: (train_dataset, test_dataset, num_classes, num_training_samples)
    train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
        args.dataset,
        args.noise_type,
        args.noise_path,
        args.is_human
    )

    # ============== 保存训练集 ==============
    # train_dataset.train_data 的形状预期是 (50000, 32, 32, 3)
    # train_dataset.train_labels 的长度为 50000
    train_images = train_dataset.train_data
    train_labels = train_dataset.train_labels
    folder_path = "cifar-10-100n/train/" + args.noise_type
    save_cifar_images_to_dir(train_images, train_labels, root_dir=folder_path)

    # ============== 是否保存测试集 ==============
    if args.save_test:
        # test_dataset.test_data, test_dataset.test_labels
        test_images = test_dataset.test_data
        test_labels = test_dataset.test_labels
        save_cifar_images_to_dir(test_images, test_labels, root_dir="test")

if __name__ == '__main__':
    main()
