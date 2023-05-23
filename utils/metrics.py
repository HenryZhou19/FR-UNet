import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None  # 当前值
        self.avg = None  # 加权平均值
        self.sum = None  # 加权求和
        self.count = None  # 总权重

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def get_metrics(predict, target, threshold=None, predict_b=None):
    """
    Args:
        predict: (N, 1, H, W)
        target: (N, 1, H, W)
        threshold: 默认 0.5 【不进行双阈值时使用】
    """
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    if predict_b is not None:
        predict_b = predict_b.flatten()  # 外部输入的结果
    else:
        predict_b = np.where(predict >= threshold, 1, 0)  # 单阈值直接获得结果
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    auc = roc_auc_score(target, predict)  # ROC 曲线和 AUC 值必定使用单阈值计算
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }


def count_connect_component(predict, target, threshold=None, connectivity=8):
    """
    Args:
        predict: (H, W) 也即 pre
        target: (H, W) 也即 gt
        threshold: 默认None, 在使用双阈值 DTI 时触发, 此时不对 predict 进行处理; 否则为 self.CFG.threshold 【高阈值】
        connectivity: 连通性定义，默认 8 邻域连通
    """
    if threshold != None:  # 对于 DTI==False 的情况，此处传入的 predict 仍是网络的输出，还需要 sigmoid 和阈值检测
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(  # 对 pre 进行 8 连通检测
        np.asarray(predict, dtype=np.uint8)*255, 
        connectivity=connectivity,
        )
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(  # 对 gt 进行 8 连通检测
        np.asarray(target, dtype=np.uint8)*255, 
        connectivity=connectivity,
        )
    return pre_n/gt_n  # 直接输出连通度比值【vessel connectivity assessment (VCA)】
