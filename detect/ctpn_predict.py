
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置CUDA设备为0号GPU
import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch中的功能性模块
from detect.ctpn_model import CTPN_Model  # 导入CTPN模型类
from detect.ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox, nms, \
    TextProposalConnectorOriented  # 导入CTPN工具函数
from detect.ctpn_utils import resize  # 导入resize函数
from detect import config  # 导入配置文件

# 设置一些参数
prob_thresh = 0.5  # 分类概率的阈值
height = 720  # 调整图像的高度
gpu = True  # 是否使用GPU
if not torch.cuda.is_available():
    gpu = False  # 如果GPU不可用，则使用CPU
device = torch.device('cuda:0' if gpu else 'cpu')  # 根据设备情况选择运行设备
weights = os.path.join(config.checkpoints_dir, 'CTPN.pth')  # 模型权重文件路径
model = CTPN_Model()  # 实例化CTPN模型
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])  # 加载预训练模型权重
model.to(device)  # 将模型移动到指定设备（GPU或CPU）
model.eval()  # 设置模型为评估模式（推理时使用）


# 显示图像的函数
def dis(image):
    cv2.imshow('image', image)  # 显示图像
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有窗口


# 获取检测框的函数
def get_det_boxes(image, display=True, expand=True):
    image = resize(image, height=height)  # 调整图像大小
    image_r = image.copy()  # 备份原始图像
    image_c = image.copy()  # 备份用于绘制的图像
    h, w = image.shape[:2]  # 获取图像的高度和宽度
    image = image.astype(np.float32) - config.IMAGE_MEAN  # 图像减去均值（归一化处理）
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()  # 转换为PyTorch张量，并调整维度顺序

    # 禁用梯度计算，提高推理速度
    with torch.no_grad():
        image = image.to(device)  # 将图像移动到指定设备
        cls, regr = model(image)  # 使用模型进行前向传播，获取分类和回归结果
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()  # 对分类结果应用softmax，并转换为NumPy数组
        regr = regr.cpu().numpy()  # 回归结果转换为NumPy数组
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)  # 生成锚点（anchors）
        bbox = bbox_transfor_inv(anchor, regr)  # 将回归结果转换为实际边界框
        bbox = clip_box(bbox, [h, w])  # 剪裁边界框，使其不超出图像边界

        # 根据分类概率阈值选择前景锚点
        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        select_anchor = bbox[fg, :]  # 选择对应的边界框
        select_score = cls_prob[0, fg, 1]  # 选择对应的分类得分
        select_anchor = select_anchor.astype(np.int32)  # 转换为整数类型
        keep_index = filter_bbox(select_anchor, 16)  # 过滤掉尺寸过小的边界框

        # 非极大值抑制（NMS）去除冗余框
        select_anchor = select_anchor[keep_index]  # 根据过滤后的索引选择边界框
        select_score = select_score[keep_index]  # 根据过滤后的索引选择得分
        select_score = np.reshape(select_score, (select_score.shape[0], 1))  # 调整得分的形状
        nmsbox = np.hstack((select_anchor, select_score))  # 将边界框和得分水平堆叠
        keep = nms(nmsbox, 0.3)  # 执行NMS，阈值为0.3
        select_anchor = select_anchor[keep]  # 根据NMS结果选择最终的边界框
        select_score = select_score[keep]  # 根据NMS结果选择最终的得分

        # 将文本提案连接成文本行
        textConn = TextProposalConnectorOriented()  # 实例化文本提案连接器
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])  # 获取最终的文本行

        # 如果启用扩展，扩展文本框的宽度
        if expand:
            for idx in range(len(text)):
                text[idx][0] = max(text[idx][0] - 10, 0)
                text[idx][2] = min(text[idx][2] + 10, w - 1)
                text[idx][4] = max(text[idx][4] - 10, 0)
                text[idx][6] = min(text[idx][6] + 10, w - 1)

        # 如果启用显示，绘制文本行和边界框
        if display:
            blank = np.zeros(image_c.shape, dtype=np.uint8)  # 创建一个空白图像
            for box in select_anchor:
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                blank = cv2.rectangle(blank, pt1, pt2, (50, 0, 0), -1)  # 在空白图像上绘制矩形
            image_c = image_c + blank  # 叠加矩形到原图像
            image_c[image_c > 255] = 255  # 将像素值限制在[0, 255]
            for i in text:
                s = str(round(i[-1] * 100, 2)) + '%'  # 生成文本行的置信度字符串
                i = [int(j) for j in i]  # 将坐标转换为整数
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)  # 绘制文本行的四条边
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                cv2.putText(image_c, s, (i[0] + 13, i[1] + 13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA)  # 在文本行上绘制置信度
        return text, image_c, image_r  # 返回检测结果，包括文本行、带有标注的图像、原始图像


# 主函数入口
if __name__ == '__main__':
    img_path = 'images/t1.png'  # 输入图像路径
    image = cv2.imread(img_path)  # 读取图像
    text, image = get_det_boxes(image)  # 获取文本检测结果
    dis(image)  # 显示检测结果
