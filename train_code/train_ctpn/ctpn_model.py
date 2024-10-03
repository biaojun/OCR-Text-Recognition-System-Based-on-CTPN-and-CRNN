


# RPN_REGR_Loss 和 RPN_CLS_Loss 类：
#
# RPN_REGR_Loss 用于计算回归损失，主要是用于预测边界框的平滑L1损失，处理的是框的位置调整。
# RPN_CLS_Loss 用于计算分类损失，主要是处理目标的分类任务，使用交叉熵损失进行正负样本的分类。
# basic_conv 类：
#
# 这是一个基础的卷积操作类，包括卷积、批量归一化和激活函数的组合。
# CTPN_Model 类：
#
# CTPN_Model 是整个模型的核心部分，继承了 PyTorch 的 nn.Module 类。
# 它首先使用 VGG16 提取图像的特征，然后通过自定义的卷积层 rpn 进一步处理。
# brnn 是双向 GRU 网络，用于从提取的特征中捕捉序列信息。
# lstm_fc 是一个全连接层，将 GRU 的输出调整为后续分类和回归分支的输入。
# 最终通过 rpn_class 和 rpn_regress 得到分类和回归的结果。

import os



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config

# 定义平滑L1损失的类，用于回归任务中的损失计算
class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        '''
        计算平滑L1损失
        :param input: 预测值 y_preds
        :param target: 真实值 y_true
        :return: 计算得到的损失
        '''
        try:
            cls = target[0, :, 0]  # 获取类别标签
            regr = target[0, :, 1:3]  # 获取回归目标
            regr_keep = (cls == 1).nonzero()[:, 0]  # 选择正样本的索引
            regr_true = regr[regr_keep]  # 获取正样本的真实回归值
            regr_pred = input[0][regr_keep]  # 获取正样本的预测回归值
            diff = torch.abs(regr_true - regr_pred)  # 计算预测值与真实值的差异
            less_one = (diff < 1.0 / self.sigma).float()  # 平滑L1损失的条件分支
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)  # 按列求和
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)  # 计算平均损失
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            loss = torch.tensor(0.0)

        return loss.to(self.device)  # 返回损失值

# 定义分类损失的类，用于分类任务中的损失计算
class RPN_CLS_Loss(nn.Module):
    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')  # 使用交叉熵损失
        self.pos_neg_ratio = 3  # 正负样本比例

    def forward(self, input, target):
        if config.OHEM:
            cls_gt = target[0][0]  # 获取分类标签
            num_pos = 0
            loss_pos_sum = 0

            if len((cls_gt == 1).nonzero()) != 0:  # 处理正样本不为0的情况
                cls_pos = (cls_gt == 1).nonzero()[:, 0]  # 获取正样本索引
                gt_pos = cls_gt[cls_pos].long()  # 获取正样本标签
                cls_pred_pos = input[0][cls_pos]  # 获取正样本的预测值
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))  # 计算正样本的损失
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)

            cls_neg = (cls_gt == 0).nonzero()[:, 0]  # 获取负样本索引
            gt_neg = cls_gt[cls_neg].long()  # 获取负样本标签
            cls_pred_neg = input[0][cls_neg]  # 获取负样本的预测值

            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))  # 计算负样本的损失
            loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), config.RPN_TOTAL_NUM - num_pos))  # 选择Top-K负样本
            loss_cls = loss_pos_sum + loss_neg_topK.sum()  # 计算总的分类损失
            loss_cls = loss_cls / config.RPN_TOTAL_NUM  # 归一化
            return loss_cls.to(self.device)  # 返回损失值
        else:
            y_true = target[0][0]  # 获取真实的分类标签
            cls_keep = (y_true != -1).nonzero()[:, 0]  # 选择非忽略的样本
            cls_true = y_true[cls_keep].long()  # 获取有效的真实标签
            cls_pred = input[0][cls_keep]  # 获取有效的预测值
            loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)  # 计算交叉熵损失
            loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)  # 计算平均损失
            return loss.to(self.device)  # 返回损失值

# 定义基本卷积层的类
class basic_conv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=True):
        super(basic_conv, self).__init__()
        self.out_channels = out_planes  # 输出通道数
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)  # 卷积层
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None  # 批量归一化层
        self.relu = nn.ReLU(inplace=True) if relu else None  # 激活函数

    def forward(self, x):
        x = self.conv(x)  # 卷积操作
        if self.bn is not None:
            x = self.bn(x)  # 批量归一化
        if self.relu is not None:
            x = self.relu(x)  # 激活函数
        return x

# 定义CTPN模型的类
class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)  # 使用VGG16作为基础模型
        layers = list(base_model.features)[:-1]  # 提取VGG16的特征提取层
        self.base_layers = nn.Sequential(*layers)  # 组合成序列化的层
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)  # 区域提议网络的卷积层
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)  # 双向GRU层
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)  # 全连接层
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)  # 分类分支
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)  # 回归分支

    def forward(self, x):
        x = self.base_layers(x)  # 提取VGG16的特征
        x = self.rpn(x)  # 区域提议网络的卷积操作
        x1 = x.permute(0, 2, 3, 1).contiguous()  # 调整通道维度顺序
        b = x1.size()  # 获取张量的尺寸信息
        x1 = x1.view(b[0] * b[1], b[2], b[3])  # 重塑张量形状以适应GRU输入
        x2, _ = self.brnn(x1)  # 通过双向GRU处理特征
        xsz = x.size()  # 保存原始尺寸
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # 恢复张量的原始尺寸
        x3 = x3.permute(0, 3, 1, 2).contiguous()  # 调整通道维度顺序
        x3 = self.lstm_fc(x3)  # 通过全连接层
        x = x3

        cls = self.rpn_class(x)  # 分类分支的输出
        regr = self.rpn_regress(x)  # 回归分支的输出
        cls = cls.permute(0, 2, 3, 1).contiguous()  # 调整通道维度顺序
        regr = regr.permute(0, 2, 3, 1).contiguous()

        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)  # 调整张量形状以适应损失计算
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)  # 调整张量形状以适应损失计算

        return cls, regr  # 返回分类和回归的结果
