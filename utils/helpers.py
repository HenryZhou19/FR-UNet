import os
import pickle
import random
import shutil
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


def dir_exists(path, must_new=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if must_new:
            raise AssertionError


def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def read_pickle(path, type):
    with open(file=path + f"/{type}.pkl", mode='rb') as file:
        img = pickle.load(file)
    return img


def save_pickle(path, type, img_list):
    with open(file=path + f"/{type}.pkl", mode='wb') as file:
        pickle.dump(img_list, file)


def double_threshold_iteration(index, img, h_thresh, l_thresh, save=True):
    """
    Args:
        index: 图像序号
        img: 实际为 pre (H, W) ,也即网络的输出结果 (未经过sigmoid) 
        h_thresh: 高阈值 [0., 1.]
        l_thresh: 低阈值 [0., 1.]
    """
    h, w = img.shape
    img = np.array(torch.sigmoid(img).cpu().detach()*255, dtype=np.uint8)  # 转换到 unint8 范围内再进行 DTI
    bin = np.where(img >= h_thresh*255, 255, 0).astype(np.uint8)  # 255 或 0
    gbin = bin.copy()  # 作为双阈值的最终结果
    gbin_pre = gbin - 1  # 254 或 255(只是为了进下面的循环)
    while(gbin_pre.all() != gbin.all()):
        # 当前一次循环内对 gbin 有改动，则再次循环
        gbin_pre = gbin  # 255 或 0
        for i in range(h):
            for j in range(w):
                # 对于每一个像素位置，如果双阈值 gbin 目前不是前景，并且其比高阈值低，比低阈值高
                if gbin[i][j] == 0 and img[i][j] < h_thresh*255 and img[i][j] >= l_thresh*255:
                    # 并且其 8 邻域内至少存在一个已经被判为前景，则将该位置也作为前景
                    if gbin[i-1][j-1] or gbin[i-1][j] or gbin[i-1][j+1] or gbin[i][j-1] or gbin[i][j+1] or gbin[i+1][j-1] or gbin[i+1][j] or gbin[i+1][j+1]:
                        gbin[i][j] = 255

    if save:
        cv2.imwrite(f"save_picture/bin{index}.png", bin)  # 单阈值的分割结果图
        cv2.imwrite(f"save_picture/gbin{index}.png", gbin)
    return gbin/255  # 转换到 float64 [0., 1.]，便于进行性能指标计算


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):  # 没有使用
    assert (len(preds.shape) == 4)
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    k = 0
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[
                    k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h,
                         w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)
    final_avg = full_prob / full_sum
    return final_avg
