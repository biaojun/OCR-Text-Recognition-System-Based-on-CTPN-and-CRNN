import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用的GPU设备为1
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import argparse

import config  # 导入自定义配置文件
from ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss  # 导入模型和损失函数
from data.dataset import ICDARDataset  # 导入数据集处理类

# 设置随机种子，保证实验可重复
random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

# 设置训练参数
epochs = 2  # 训练的轮次
lr = 1e-3  # 学习率
resume_epoch = 0  # 从第几个epoch恢复训练


# 定义保存模型检查点的函数
def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    check_path = os.path.join(config.checkpoints_dir,  # 设置保存路径
                              f'v3_ctpn_ep{epoch:02d}_'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')

    try:
        torch.save(state, check_path)  # 尝试保存模型状态字典
    except BaseException as e:
        print(e)
        print('fail to save to {}'.format(check_path))
    print('saving to {}'.format(check_path))


# 初始化模型权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # 对卷积层进行初始化
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # 对批量归一化层进行初始化
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    # print(torch.cuda.is_available())  # 检查CUDA是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设置设备为GPU1
    checkpoints_weight = config.pretrained_weights  # 获取预训练权重路径
    print('exist pretrained ', os.path.exists(checkpoints_weight))  # 检查预训练权重是否存在
    print(torch.cuda.is_available())

    if os.path.exists(checkpoints_weight):
        pretrained = False  # 如果存在预训练权重，则不进行模型权重初始化

    # 初始化数据集和数据加载器
    dataset = ICDARDataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)

    # 初始化模型并加载到设备上
    model = CTPN_Model()
    model.to(device)

    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)  # 加载预训练模型权重
        model.load_state_dict(cc['model_state_dict'])  # 加载模型的状态字典
        resume_epoch = cc['epoch']  # 恢复训练的起始轮次
    else:
        model.apply(weights_init)  # 如果没有预训练权重，则对模型进行初始化

    params_to_uodate = model.parameters()  # 获取模型的所有参数
    optimizer = optim.SGD(params_to_uodate, lr=lr, momentum=0.9)  # 设置优化器为SGD

    # 初始化损失函数
    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)

    # 初始化记录最佳损失的变量
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch  # 调整总训练轮次
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率调度器

    # 开始训练循环
    for epoch in range(resume_epoch + 1, epochs):
        print(f'Epoch {epoch}/{epochs}')  # 打印当前训练轮次
        print('#' * 50)
        epoch_size = len(dataset) // 1  # 计算每轮次的batch数
        model.train()  # 设置模型为训练模式
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)  # 更新学习率

        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            # 数据转移到设备上
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)

            optimizer.zero_grad()  # 梯度清零

            # 前向传播
            out_cls, out_regr = model(imgs)
            # 计算损失
            loss_cls = critetion_cls(out_cls, clss)
            loss_regr = critetion_regr(out_regr, regrs)

            loss = loss_cls + loss_regr  # 总损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            # 累计损失
            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i + 1

            # 打印每个batch的损失
            print(f'Ep:{epoch}/{epochs - 1}--'
                  f'Batch:{batch_i}/{epoch_size}\n'
                  f'batch: loss_cls:{loss_cls.item():.4f}--loss_regr:{loss_regr.item():.4f}--loss:{loss.item():.4f}\n'
                  f'Epoch: loss_cls:{epoch_loss_cls / mmp:.4f}--loss_regr:{epoch_loss_regr / mmp:.4f}--'
                  f'loss:{epoch_loss / mmp:.4f}\n')

        # 计算每轮次的平均损失
        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print(f'Epoch:{epoch}--{epoch_loss_cls:.4f}--{epoch_loss_regr:.4f}--{epoch_loss:.4f}')

        # 如果当前模型损失小于之前的最佳损失，则保存模型
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(),
                             'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)

    # 训练结束后清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
