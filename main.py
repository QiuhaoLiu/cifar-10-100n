# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse #argparse 模块，用于解析命令行参数
import pathlib


##命令行参数，可以动态地修改学习率、噪声类型、数据集选择、训练次数、子进程数量等
parser = argparse.ArgumentParser() #argparse,命令行解析模块，手动调整学习率等
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='noisy100')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar100')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading') #并行子进程的数目
parser.add_argument('--is_human', action='store_true', default=False)

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan): #调整优化器的学习率，前60个学习率为0.1，后40个学习率为0.01
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    ##模型预测的前k个类别中，是否包含真实标签
    output = F.softmax(logit, dim=1) ##通过softmax函数，把原始的预测分数转化到[0,1]范围内
    print(output[:3]) # 把logit经过softmax后，得到的预测概率就是我们想要的prediction probs
    #logit.shape = torch.Size([128, 10])
    maxk = max(topk) ##
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) ##调用Pytorch的topk函数，返回前5个最大值的索引，按照降序排列
    ##values, pred = output.topk(2, 1, True, True)
    pred = pred.t() ##转置
    correct = pred.eq(target.reshape(1, -1).expand_as(pred)) ##判断pred，是否与目标的真实标签相等，比较前，将真实标签扩展到与预测结果pred形状一致

    res = []
    ##计算topk精度
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose() ##将索引转化为numpy格式，并转置
        batch_size = len(ind)
       
        images = Variable(images).cuda() ##将数据转化为pytorch的variable，支持自动的梯度计算
        labels = Variable(labels).cuda()
       
        # Forward + Backward + Optimize
        logits = model(images) ##将当前批次的图像输入模型，得到未归一化的输出结果logits,logtis是tensor类型的变量

        prec, _ = accuracy(logits, labels, topk=(1, 5)) ##计算出当前批次top1的准确率，也就是取概率最高的那个，与真实标签做对比，看看一个批次里，的预测准确率有多少
        # prec = 0.0
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad() ##梯度归零
        loss.backward() ##反向传播，计算梯度
        optimizer.step() ##更新参数
        if (i+1) % args.print_freq == 0: ##每隔args.print_freq个批次，打印一次训练精度和损失函数
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))


    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    #batchnorm是什么意思？
    #数据被按照批量归一化，使得神经网络便于训练
    #best_acc_ = argmax
    #print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc



#####################################main code ################################################
if __name__ == '__main__':

    args = parser.parse_args() ##通过args，访问命令行输入的值

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) #随机种子保存随机数据的顺序，保证结果的可复现性

    # 其他初始化代码
    batch_size = 128
    learning_rate = args.lr
    noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                      'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    args.noise_type = noise_type_map[args.noise_type]

    # 加载数据集\data
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './cifar-10-100n/data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './cifar-10-100n/data/CIFAR-100_human.pt'
        else:
            raise NameError(f'Undefined dataset {args.dataset}')

    train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset, args.noise_type,
                                                                                   args.noise_path, args.is_human)
    noise_prior = train_dataset.noise_prior
    noise_or_not = train_dataset.noise_or_not
    print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10]) ##训练集的前十个标签

    # 加载模型
    print('building model...')
    model = ResNet34(num_classes)
    print('building model done')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9) ##优化器SGD，随机梯度下降
    ##weight_decay 权重衰减，避免模型参数过大导致过拟合
    ##momentum 动量因子

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               num_workers=args.num_workers,
                                               shuffle=True)
    #在训练集中，随机打乱数据的顺序，防止模型过拟合，学习到模型的固定顺序，增强泛化能力
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              num_workers=args.num_workers,
                                              shuffle=False)
    #在测试集中，只需要评估模型性能，不需要打乱数据顺序
    alpha_plan = [0.1] * 60 + [0.01] * 40 #alpha_plan，前60个为0.1，后40个是0.01
    model.cuda()
    # 开始训练
    for epoch in range(args.n_epoch):
        print(f'epoch {epoch}')
        adjust_learning_rate(optimizer, epoch, alpha_plan)
        model.train()##模型为训练模式，模型会随机的dropout（根据设计的丢弃概率p)以防止过拟合 #而在model.eval模式下，dropout会被停用，所有神经元都会被激活，用于评估
        #
        #训练模式下，需要反向传播更新参数；而评估模式下，只用前向传播
        train_acc = train(epoch, train_loader, model, optimizer)
        # evaluate models
        test_acc = evaluate(test_loader=test_loader, model=model)
        # 保存结果
        print('train acc on train images is ', train_acc)
        print('test acc on test images is ', test_acc)
    