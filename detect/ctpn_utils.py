
import numpy as np
import cv2
from detect.config import *


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    调整图像大小，保持宽高比。如果未指定 width 和 height，则返回原图像。

    Parameters:
    - image: 输入的图像。
    - width: 目标宽度（如果提供）。
    - height: 目标高度（如果提供）。
    - inter: 插值方法，默认为 cv2.INTER_AREA，适用于缩小图像。

    Returns:
    - resized: 调整大小后的图像。
    """
    dim = None
    (h, w) = image.shape[:2]  # 获取图像的高度和宽度

    # 如果宽度和高度都未指定，返回原图像
    if width is None and height is None:
        return image

    # 如果未指定宽度，根据高度计算宽高比
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # 如果未指定高度，根据宽度计算宽高比
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # 调整图像大小，保持宽高比
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def gen_anchor(featuresize, scale):
    """
    根据特征图生成锚点（anchors）。

    Parameters:
    - featuresize: 特征图的大小（高度和宽度）。
    - scale: 缩放比例。

    Returns:
    - anchor: 生成的锚点数组，形状为 (num_anchors, 4)，每个锚点由 [x1, y1, x2, y2] 表示。
    """
    # 锚点的高度和宽度列表
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # 生成 k=9 个锚点的尺寸 (h, w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchor = np.array([0, 0, 15, 15])  # 基础锚点 [x1, y1, x2, y2]
    xt = (base_anchor[0] + base_anchor[2]) * 0.5  # 中心点 x
    yt = (base_anchor[1] + base_anchor[3]) * 0.5  # 中心点 y

    # 计算每个锚点的四个坐标
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    # 根据特征图大小生成网格，并应用偏移生成所有锚点
    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))


def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    计算两个边界框（box）之间的交并比（IoU）。

    Parameters:
    - box1: 第一个边界框 [x1, y1, x2, y2]。
    - box1_area: 第一个边界框的面积。
    - boxes2: 第二组边界框 [Msample, x1, y1, x2, y2]。 全部边界框
    - boxes2_area: 第二组边界框的面积。

    Returns:
    - iou: 交并比（IoU）数组。
    """
    # 计算两个边界框的交集部分
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)  # 交集面积
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])  # 交并比（IoU）
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    计算两个集合中边界框之间的所有交并比（IoU）。

    Parameters:
    - boxes1: 第一组边界框 [Nsample, x1, y1, x2, y2]。
    - boxes2: 第二组边界框 [Msample, x1, y1, x2, y2]。

    Returns:
    - overlaps: 交并比（IoU）矩阵。
    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])  # 第一组边界框的面积
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])  # 第二组边界框的面积

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))  # 初始化交并比矩阵

    # 计算 boxes1（锚点）和 boxes2（GT box）的交并比
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps


def bbox_transfrom(anchors, gtboxes):
    """
    计算相对锚点位置的预测垂直坐标 Vc 和 Vh。

    Parameters:
    - anchors: 锚点 [Nsample, x1, y1, x2, y2]。
    - gtboxes: 实际标注框 [Msample, x1, y1, x2, y2]。

    Returns:
    - regr: 预测框的垂直坐标变换 [Vc, Vh]。
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5  # GT box 的中心 y 坐标
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5  # 锚点的中心 y 坐标
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0  # GT box 的高度
    ha = anchors[:, 3] - anchors[:, 1] + 1.0  # 锚点的高度

    Vc = (Cy - Cya) / ha  # 垂直中心坐标的相对变换
    Vh = np.log(h / ha)  # 高度的相对变换

    return np.vstack((Vc, Vh)).transpose()


def bbox_transfor_inv(anchor, regr):
    """
    将预测的垂直坐标变换还原为实际的边界框。

    Parameters:
    - anchor: 锚点 [Nsample, x1, y1, x2, y2]。
    - regr: 预测的变换参数 [Vc, Vh]。

    Returns:
    - bbox: 预测的边界框 [x1, y1, x2, y2]。
    """
    Cya = (anchor[:, 1] + anchor[:, 3]) * 0.5  # 锚点的中心 y 坐标
    ha = anchor[:, 3] - anchor[:, 1] + 1  # 锚点的高度

    Vcx = regr[0, :, 0]  # 预测的垂直中心坐标变换
    Vhx = regr[0, :, 1]  # 预测的高度变换

    Cyx = Vcx * ha + Cya  # 还原后的中心 y 坐标
    hx = np.exp(Vhx) * ha  # 还原后的高度
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5  # 锚点的中心 x 坐标

    # 计算还原后的边界框坐标
    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


def clip_box(bbox, im_shape):
    """
    将边界框限制在图像范围内，确保边界框不超出图像边界。

    Parameters:
    - bbox: 输入的边界框 [Nsample, x1, y1, x2, y2]。
    - im_shape: 图像的尺寸 [高度, 宽度]。

    Returns:
    - bbox: 限制在图像范围内的边界框。
    """
    # 确保边界框的左上角和右下角坐标在图像范围内
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox, minsize):
    """
    过滤掉尺寸小于最小值的边界框。

    Parameters:
    - bbox: 输入的边界框 [Nsample, x1, y1, x2, y2]。
    - minsize: 最小尺寸。

    Returns:
    - keep: 过滤后保留的边界框索引。
    """
    ws = bbox[:, 2] - bbox[:, 0] + 1  # 计算边界框的宽度
    hs = bbox[:, 3] - bbox[:, 1] + 1  # 计算边界框的高度
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]  # 保留大于最小尺寸的边界框
    return keep


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    """
    计算RPN（区域建议网络）的目标标签和回归目标。

    Parameters:
    - imgsize: 原始图像的尺寸 [高度, 宽度]。
    - featuresize: 特征图的尺寸 [高度, 宽度]。
    - scale: 缩放比例。
    - gtboxes: 真实标注框 [Msample, x1, y1, x2, y2]。

    Returns:
    - labels: RPN的标签，1表示正样本，0表示负样本，-1表示忽略。
    - bbox_targets: RPN的回归目标。
    - base_anchor: 生成的基础锚点。
    """
    imgh, imgw = imgsize

    # 生成基础锚点
    base_anchor = gen_anchor(featuresize, scale)

    # 计算交并比（IoU）
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # 初始化标签，-1 表示忽略，0 表示负样本，1 表示正样本
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    # 为每个 GT box 分配一个与其 IoU 最大的锚点
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # 为每个锚点分配一个与其 IoU 最大的 GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IoU > IOU_POSITIVE 的锚点被标记为正样本
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    # IoU < IOU_NEGATIVE 的锚点被标记为负样本
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0
    # 确保每个 GT box 至少有一个正样本锚点
    labels[gt_argmax_overlaps] = 1

    # 仅保留在图像范围内的锚点
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # 正样本数量大于 RPN_POSITIVE_NUM（默认128）时进行随机采样
    fg_index = np.where(labels == 1)[0]
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

    # 负样本数量大于 RPN_TOTAL_NUM - 正样本数量 时进行随机采样
    bg_index = np.where(labels == 0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
    if (len(bg_index) > num_bg):
        labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # 计算边界框回归目标
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets], base_anchor


def nms(dets, thresh):
    """
    非极大值抑制（NMS）算法，用于去除冗余的重叠边界框。

    Parameters:
    - dets: 检测到的边界框，形状为 [Nsample, 5]，其中最后一列是得分。
    - thresh: IoU 阈值，决定是否保留某个边界框。

    Returns:
    - keep: NMS后保留的边界框索引。
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算每个边界框的面积
    order = scores.argsort()[::-1]  # 根据得分从高到低排序

    keep = []
    while order.size > 0:
        i = order[0]  # 选择得分最高的边界框
        keep.append(i)  # 保留这个边界框
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 计算与其他边界框的交集
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h  # 交集面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算IoU

        inds = np.where(ovr <= thresh)[0]  # 保留IoU小于阈值的边界框
        order = order[inds + 1]
    return keep


# 用于预测的图结构类
class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        """
        获取图中所有连接的子图。

        Returns:
        - sub_graphs: 所有连接的子图列表。
        """
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


# 文本行配置类
class TextLineCfg:
    SCALE = 600  # 缩放比例
    MAX_SCALE = 1200  # 最大缩放比例
    TEXT_PROPOSALS_WIDTH = 16  # 文本提案的宽度
    MIN_NUM_PROPOSALS = 2  # 最小提案数量
    MIN_RATIO = 0.5  # 最小长宽比
    LINE_MIN_SCORE = 0.9  # 最小行得分
    MAX_HORIZONTAL_GAP = 60  # 最大水平间隔
    TEXT_PROPOSALS_MIN_SCORE = 0.7  # 最小文本提案得分
    TEXT_PROPOSALS_NMS_THRESH = 0.3  # 文本提案的NMS阈值
    MIN_V_OVERLAPS = 0.6  # 最小垂直重叠
    MIN_SIZE_SIM = 0.6  # 最小尺寸相似度


# 文本提案图构建器类
class TextProposalGraphBuilder:
    """
    将文本提案构建为图结构。
    """

    def get_successions(self, index):
        """
        获取指定节点的后续节点列表。

        Parameters:
        - index: 当前节点索引。

        Returns:
        - results: 后续节点索引列表。
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        """
        获取指定节点的前驱节点列表。

        Parameters:
        - index: 当前节点索引。

        Returns:
        - results: 前驱节点索引列表。
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        """
        判断节点 succession_index 是否是 index 的后续节点。

        Parameters:
        - index: 当前节点索引。
        - succession_index: 后续节点索引。

        Returns:
        - bool: 如果是后续节点，则返回True。
        """
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        """
        判断两个节点是否满足垂直方向上的IoU和尺寸相似度。

        Parameters:
        - index1: 第一个节点索引。
        - index2: 第二个节点索引。

        Returns:
        - bool: 如果满足条件，则返回True。
        """

        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        """
        构建文本提案的图结构。

        Parameters:
        - text_proposals: 文本提案（边界框）。
        - scores: 每个文本提案的得分。
        - im_size: 图像的尺寸。

        Returns:
        - Graph: 构建好的图结构。
        """
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True
        return Graph(graph)


# 连接文本提案成文本行的类
class TextProposalConnectorOriented:
    """
    将文本提案连接成文本行。
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        """
        将文本提案分组并生成文本行。

        Parameters:
        - text_proposals: 文本提案（边界框）。
        - scores: 每个文本提案的得分。
        - im_size: 图像的尺寸。

        Returns:
        - list: 每个文本行的边界框索引列表。
        """
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        """
        进行线性拟合，生成文本行的上下边界线。

        Parameters:
        - X: x坐标数组。
        - Y: y坐标数组。
        - x1: 左边界的x坐标。
        - x2: 右边界的x坐标。

        Returns:
        - tuple: 拟合线在x1和x2处的y坐标。
        """
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        获取最终的文本行。

        Parameters:
        - text_proposals: 文本提案（边界框）。
        - scores: 每个文本提案的得分。
        - im_size: 图像的尺寸。

        Returns:
        - np.ndarray: 每个文本行的边界框 [x1, y1, x2, y2, score, k, b, height]。
        """
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 每个文本行的所有小框
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 每个小框的中心x坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2  # 每个小框的中心y坐标

            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据中心点拟合一条直线

            x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

            # 拟合文本行上下边界线，计算左边界和右边界对应的y坐标
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 计算文本行的平均得分

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上边界线的最小y坐标
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下边界线的最大y坐标
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 中心线的斜率k
            text_lines[index, 6] = z1[1]  # 中心线的截距b
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 计算文本行的平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 计算文本行上边界线的截距
            b2 = line[6] + line[7] / 2  # 计算文本行下边界线的截距
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上角点
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上角点
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下角点
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下角点
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 计算文本行的宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 高度补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]  # 文本行得分
            index = index + 1

        return text_recs  # 返回文本行的最终边界框
