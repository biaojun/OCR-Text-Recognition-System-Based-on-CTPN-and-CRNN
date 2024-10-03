# 导入必要的库
import os  # 导入操作系统接口模块
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的函数接口模块
import torchvision.models as models  # 导入 torchvision 的预训练模型模块

# 定义回归损失类
class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma  # 初始化参数 sigma，用于平滑 L1 损失
        self.device = device  # 初始化参数 device，表示计算设备（如 CPU 或 GPU）

    def forward(self, input, target):
        '''
        smooth L1 loss
        :param input: y_preds
        :param target: y_true
        :return: loss
        定义和实现了smoth_L1
        '''
        try:
            cls = target[0, :, 0]  # 提取目标中的分类标签
            regr = target[0, :, 1:3]  # 提取目标中的回归标签
            regr_keep = (cls == 1).nonzero()[:, 0]  # 找到分类标签为 1 的位置（即正样本）
            regr_true = regr[regr_keep]  # 提取正样本的回归真实值
            regr_pred = input[0][regr_keep]  # 提取正样本的回归预测值
            diff = torch.abs(regr_true - regr_pred)  # 计算预测值与真实值之间的绝对差异
            less_one = (diff < 1.0 / self.sigma).float()  # 找到差异小于阈值的元素位置
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)  # 计算平滑 L1 损失
            loss = torch.sum(loss, 1)  # 按列求和损失
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)  # 计算损失均值，如果没有损失值，则返回 0
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            loss = torch.tensor(0.0)  # 捕获异常，打印异常信息并返回损失 0

        return loss.to(self.device)  # 返回损失值，并将其移动到指定设备（CPU 或 GPU）

# 定义分类损失类
class RPN_CLS_Loss(nn.Module):
    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device  # 初始化参数 device，表示计算设备（如 CPU 或 GPU）

    def forward(self, input, target):
        y_true = target[0][0]  # 提取目标中的分类标签
        cls_keep = (y_true != -1).nonzero()[:, 0]  # 找到不为 -1 的位置（即有效样本）
        cls_true = y_true[cls_keep].long()  # 提取有效样本的分类真实值
        cls_pred = input[0][cls_keep]  # 提取有效样本的分类预测值
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)  # 计算负对数似然损失（NLL Loss）
        loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)  # 计算损失均值并限制在 0 到 10 之间，如果没有损失值，则返回 0
        return loss.to(self.device)  # 返回损失值，并将其移动到指定设备（CPU 或 GPU）

# 定义基本卷积模块
class basic_conv(nn.Module):
    def __init__(self,
                 in_planes,  # 输入通道数
                 out_planes,  # 输出通道数
                 kernel_size,  # 卷积核大小
                 stride=1,  # 步幅，默认值为 1
                 padding=0,  # 填充，默认值为 0
                 dilation=1,  # 扩展，默认值为 1
                 groups=1,  # 组，默认值为 1
                 relu=True,  # 是否使用 ReLU 激活函数，默认值为 True
                 bn=True,  # 是否使用批量归一化，默认值为 True
                 bias=True):  # 是否使用偏置，默认值为 True
        super(basic_conv, self).__init__()
        self.out_channels = out_planes  # 设置输出通道数
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)  # 定义卷积层
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None  # 定义批量归一化层（如果使用）
        self.relu = nn.ReLU(inplace=True) if relu else None  # 定义 ReLU 激活函数（如果使用）

    def forward(self, x):
        x = self.conv(x)  # 应用卷积层
        if self.bn is not None:
            x = self.bn(x)  # 应用批量归一化层（如果使用）
        if self.relu is not None:
            x = self.relu(x)  # 应用 ReLU 激活函数（如果使用）
        return x

# 定义 CTPN 模型
class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)  # 加载 VGG16 模型，不使用预训练权重
        layers = list(base_model.features)[:-1]  # 获取 VGG16 的特征层，去掉最后一个卷积层
        self.base_layers = nn.Sequential(*layers)  # 将特征层转换为一个顺序容器
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)  # 定义 RPN 层
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)  # 定义双向 GRU 层 GRU (Gated Recurrent Unit)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)  # 定义 LSTM 后的全连接卷积层
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)  # 定义 RPN 分类层
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)  # 定义 RPN 回归层
    def forward(self, x):
        x = self.base_layers(x)  # 输入图像经过基础层（VGG16 的特征层）
        x = self.rpn(x)  # 经过 RPN 层生成特征图

        x1 = x.permute(0, 2, 3, 1).contiguous()  # 转换特征图的维度，将通道移到最后一维
        b = x1.size()  # 获取转换后的特征图尺寸
        x1 = x1.view(b[0] * b[1], b[2], b[3])  # 重新调整特征图的尺寸，准备输入到双向 GRU

        x2, _ = self.brnn(x1)  # 经过双向 GRU 层

        xsz = x.size()  # 获取原始特征图的尺寸
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # 重新调整 GRU 输出的尺寸

        x3 = x3.permute(0, 3, 1, 2).contiguous()  # 转换特征图的维度，将通道移回第一维
        x3 = self.lstm_fc(x3)  # 经过 LSTM 后的全连接卷积层
        x = x3

        cls = self.rpn_class(x)  # 经过 RPN 分类层
        regr = self.rpn_regress(x)  # 经过 RPN 回归层

        cls = cls.permute(0, 2, 3, 1).contiguous()  # 转换分类结果的维度
        regr = regr.permute(0, 2, 3, 1).contiguous()  # 转换回归结果的维度

        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)  # 调整分类结果的尺寸
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)  # 调整回归结果的尺寸

        return cls, regr  # 返回分类和回归结果
