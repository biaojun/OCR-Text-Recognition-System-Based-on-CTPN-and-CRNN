import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #指定使用的GPU设备为1
import torch
from torch.utils.data import DataLoader
from torch import optim
import  numpy as np
import argparse

import config  # 导入自定义配置文件
from ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss  # 导入模型和损失函数
from data.dataset import ICDARDataset  # 导入数据集处理类

random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

epochs = 2
lr = 1e-3
resume_epoch = 0

def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext = 'pth'):
    check_path = os.path.join(config.checkpoints_dir,
                              f'v3_ctpn_ep{epoch:02d}_'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')
    try:
        torch.save(state, check_path)
    except BaseException as e:
        print(e)
        print("fail to save to".format(check_path))
    print('saving to {}'.format(check_path))


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    chekpoints_weight = config.pretrained_weights

    if os.path.exists(chekpoints_weight):
        pretrained = False

    #导入数据集
    dataset = ICDARDataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=config.num_workers)

    model = CTPN_Model()
    model.to(device)

    if os.path.exists(chekpoints_weight):
        print('using pretrained weight:{}'.format(chekpoints_weight))
        cc = torch.load(chekpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weight_init())

    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, lr = lr, momentum=0.9)

    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)

    #初始化记录最佳损失函数变量
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch
    #设置学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #训练循环
    for epoch in range(resume_epoch, epochs):
        print(f'Epoch{epoch}/{epochs}')
        print('*' * 50)
        epoch_size = len(dataset) // 1
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)

        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)

            optimizer.zero_grad()

            out_cls, out_regr = model(imgs)
            #分类损失为 是否为正样本或者为负样本
            loss_cls = critetion_cls(out_cls, clss)
            loss_regr = critetion_regr(out_regr, regrs)
            loss = loss_cls + loss_regr
            loss.backward()

            #优化器更新权重，跟学习率有关
            optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i + 1



