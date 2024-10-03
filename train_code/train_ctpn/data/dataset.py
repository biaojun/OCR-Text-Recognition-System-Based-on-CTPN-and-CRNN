import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from ..config import IMAGE_MEAN
from ..ctpn_utils import cal_rpn




def readxml(path):
    """
    读取XML文件，提取图像的文件名和标注的边界框（ground truth boxes）。

    参数:
    - path: XML文件的路径。

    返回:
    - gtboxes: 包含边界框的数组，格式为 [xmin, ymin, xmax, ymax]。
    - imgfile: 图像文件名。
    """
    gtboxes = []
    imgfile = ''
    xml = ET.parse(path)  # 解析XML文件
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text  # 获取图像文件名
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    # 获取边界框的坐标
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))

                    gtboxes.append((xmin, ymin, xmax, ymax))  # 将边界框添加到列表中

    return np.array(gtboxes), imgfile  # 返回边界框数组和图像文件名

# 用于CTPN文本检测的自定义数据集类
class VOCDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        """
        初始化数据集。

        参数:
        - datadir: 图像文件的目录。
        - labelsdir: 标注文件的目录（XML格式）。
        """
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir  # 图像文件目录
        self.img_names = os.listdir(self.datadir)  # 图像文件名列表
        self.labelsdir = labelsdir  # 标注文件目录

    def __len__(self):
        """返回数据集的大小，即图像的数量。"""
        return len(self.img_names)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的图像及其对应的标注。

        参数:
        - idx: 图像和标注的索引。

        返回:
        - m_img: 预处理后的图像数据。
        - cls: 目标类别标签。
        - regr: 回归目标。
        """
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)  # 获取图像路径
        print(img_path)
        xml_path = os.path.join(self.labelsdir, img_name.replace('.jpg', '.xml'))  # 获取对应的XML文件路径
        gtbox, _ = readxml(xml_path)  # 读取XML文件，获取标注的边界框
        img = cv2.imread(img_path)  # 读取图像文件
        h, w, c = img.shape  # 获取图像的尺寸

        # 随机翻转图像
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]  # 左右翻转图像
            newx1 = w - gtbox[:, 2] - 1  # 翻转后的xmin
            newx2 = w - gtbox[:, 0] - 1  # 翻转后的xmax
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        # 计算RPN的类别标签和回归目标
        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)

        m_img = img - IMAGE_MEAN  # 减去图像均值进行归一化

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])  # 将类别标签和回归目标合并

        cls = np.expand_dims(cls, axis=0)  # 扩展维度以匹配网络输入要求

        # 将数据转换为PyTorch张量
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()  # 转换图像维度为 (C, H, W)
        cls = torch.from_numpy(cls).float()  # 转换类别标签为张量
        regr = torch.from_numpy(regr).float()  # 转换回归目标为张量

        return m_img, cls, regr  # 返回图像数据、类别标签和回归目标

class ICDARDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        """
        初始化ICDAR数据集。

        参数:
        - datadir: 图像文件的目录。
        - labelsdir: 标注文件的目录。
        """
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir  # 图像文件目录
        self.img_names = os.listdir(self.datadir)  # 图像文件名列表
        self.labelsdir = labelsdir  # 标注文件目录

    def __len__(self):
        """返回数据集的大小，即图像的数量。"""
        return len(self.img_names)

    def box_transfer(self, coor_lists, rescale_fac=1.0):
        """
        将坐标列表转换为边界框列表。

        参数:
        - coor_lists: 坐标列表。
        - rescale_fac: 缩放因子。

        返回:
        - gtboxes: 边界框数组，格式为 [xmin, ymin, xmax, ymax]。
        """
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            gtboxes.append((xmin, ymin, xmax, ymax))
        return np.array(gtboxes)  # 返回边界框数组

    def box_transfer_v2(self, coor_lists, rescale_fac=1.0):
        """
        将坐标列表转换为边界框列表（v2版本）。

        参数:
        - coor_lists: 坐标列表。
        - rescale_fac: 缩放因子。

        返回:
        - gtboxes: 边界框数组，格式为 [xmin, ymin, xmax, ymax]。
        """
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16 * i - 0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)  # 返回分割后的边界框数组

    def parse_gtfile(self, gt_path, rescale_fac=1.0):
        """
        解析标注文件，获取边界框列表。

        参数:
        - gt_path: 标注文件的路径。
        - rescale_fac: 缩放因子。

        返回:
        - 解析后的边界框数组。
        """
        coor_lists = list()
        with open(gt_path, encoding='UTF-8') as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(',')[:8]
                if len(coor_list) == 8:
                    coor_lists.append(coor_list)
        return self.box_transfer_v2(coor_lists, rescale_fac)  # 调用box_transfer_v2方法

    def draw_boxes(self, img, cls, base_anchors, gt_box):
        """
        在图像上绘制边界框。

        参数:
        - img: 原始图像。
        - cls: 类别标签。
        - base_anchors: 基本锚框。
        - gt_box: 真实边界框。

        返回:
        - 绘制了边界框的图像。
        """
        for i in range(len(cls)):
            if cls[i] == 1:
                pt1 = (int(base_anchors[i][0]), int(base_anchors[i][1]))
                pt2 = (int(base_anchors[i][2]), int(base_anchors[i][3]))
                img = cv2.rectangle(img, pt1, pt2, (200, 100, 100))  # 以红色框绘制正样本的边界框
        for i in range(gt_box.shape[0]):
            pt1 = (int(gt_box[i][0]), int(gt_box[i][1]))
            pt2 = (int(gt_box[i][2]), int(gt_box[i][3]))
            img = cv2.rectangle(img, pt1, pt2, (100, 200, 100))  # 以绿色框绘制真实边界框
        return img  # 返回带有边界框的图像

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的图像及其对应的标注。

        参数:
        - idx: 图像和标注的索引。

        返回:
        - m_img: 预处理后的图像数据。
        - cls: 目标类别标签。
        - regr: 回归目标。
        """
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)  # 获取图像路径
        img = cv2.imread(img_path)  # 读取图像文件

        # 处理图像读取错误，使用默认图像替代
        if img is None:
            print(img_path)
            with open('error_imgs.txt', 'a') as f:
                f.write('{}\n'.format(img_path))
            img_name = 'img_2647.jpg'
            img_path = os.path.join(self.datadir, img_name)
            img = cv2.imread(img_path)

        h, w, c = img.shape  # 获取图像的尺寸
        rescale_fac = max(h, w) / 1600  # 计算缩放因子
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img, (w, h))  # 调整图像大小

        gt_path = os.path.join(self.labelsdir, 'gt_' + img_name.split('.')[0] + '.txt')
        gtbox = self.parse_gtfile(gt_path, rescale_fac)  # 解析标注文件，获取边界框

        # 随机翻转图像
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]  # 左右翻转图像
            newx1 = w - gtbox[:, 2] - 1  # 翻转后的xmin
            newx2 = w - gtbox[:, 0] - 1  # 翻转后的xmax
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)  # 计算RPN的类别标签和回归目标

        m_img = img - IMAGE_MEAN  # 减去图像均值进行归一化

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])  # 将类别标签和回归目标合并

        cls = np.expand_dims(cls, axis=0)  # 扩展维度以匹配网络输入要求

        # 将数据转换为PyTorch张量
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()  # 转换图像维度为 (C, H, W)
        cls = torch.from_numpy(cls).float()  # 转换类别标签为张量
        regr = torch.from_numpy(regr).float()  # 转换回归目标为张量

        return m_img, cls, regr  # 返回图像数据、类别标签和回归目标
