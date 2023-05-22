import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import safe_load
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files, remove_dir


def data_process(data_path, name, patch_size, stride, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")  # 用来以pkl格式存储处理后的数据（train 会切为 patches，test不变）
    remove_dir(save_path)
    dir_exists(save_path,
               #    must_new=True,
               )
    # remove_files(save_path)  # 上一行要求该文件夹本身不存在，是新创建的，这一行又额外删除该文件夹的内容（没必要）
    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        file_list = list(sorted(os.listdir(data_path)))  # 对于CHASEDB1数据集，此处直接得到所有图像文件名（包括img和gt）
    '''
    elif name == "STARE":
        img_path = os.path.join(data_path, "stare-images")
        gt_path = os.path.join(data_path, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    '''
    img_list = []
    gt_list = []
    rgb2gray = Grayscale(1)
    img2tensor = ToTensor()
    for i, file in enumerate(file_list):
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = rgb2gray(img)  # 图像转灰度
            img_list.append(img2tensor(img))
            gt_list.append(img2tensor(gt))

        elif name == "CHASEDB1":
            if len(file) == 13:  # 循环过程仅对 img 文件进行，gt 与之对应
                if mode == "training" and int(file[6:8]) <= 10:  # PIL(960, 999, 3)---[0, 255]
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))  # PIL(960, 999)---[0, 255]
                    img = rgb2gray(img)  # PIL(960, 999) [0, 255]
                    img_list.append(img2tensor(img))  # [tensor(1, 960, 999) , ...]---[0., 1.]
                    gt_list.append(img2tensor(gt))  # [tensor(1, 960, 999) , ...]
                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = rgb2gray(img)
                    img_list.append(img2tensor(img))
                    gt_list.append(img2tensor(gt))
        '''
        elif name == "DCA1":
            if len(file) <= 7:
                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        '''
    img_list = normalization(img_list)  # [tensor(1, H, W) , ...]
    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)  # [tensor(1, patch_size, patch_size) , ...]
        gt_patch = get_patch(gt_list, patch_size, stride)  # [tensor(1, patch_size, patch_size) , ...]
        save_imgs_pickle(img_patch, save_path, "img_patch", name)  # name 是数据集名称
        save_imgs_pickle(gt_patch, save_path, "gt_patch", name)
    elif mode == "test":
        if name != "CHUAC":
            img_list = get_square(img_list, name) # [tensor(1, shape_pad, shape_pad) , ...]
            gt_list = get_square(gt_list, name) # [tensor(1, shape_pad, shape_pad) , ...]
        save_imgs_pickle(img_list, save_path, "img", name)
        save_imgs_pickle(gt_list, save_path, "gt", name)


def get_square(img_list, name):
    img_s = []
    # FR-UNet 需要至少 3 次 2×2 下采样，此处将原始图像填充至 16 的倍数且为正方形（需要验证是否是模型内部存在16倍下采样）
    if name == "DRIVE":  # (H=584, W=565)
        shape = 592
    elif name == "CHASEDB1":  # (H=960, W=999)
        shape = 1008
    elif name == "DCA1":
        shape = 320
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)  # 在原图右下角使用 0 填充
    for i in range(len(img_list)):
        img = pad(img_list[i])  # tensor(1, H, W) → tensor(1, shape_pad, shape_pad)
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size=48, stride=6):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride  # 图像不一定是 stride=6 的整数倍，因此计算填充量
    pad_w = stride - (w - patch_size) % stride
    pad_h = pad_h if pad_h < stride else 0  # 上述实现有问题，如果本身是整数倍，则还会多pad 6像素---此处进行修改
    pad_w = pad_w if pad_w < stride else 0
    # num_H = (h + pad_h - patch_size) / stride + 1 即为 H 方向能切分的 patches 数量
    # num_W = (w + pad_w - patch_size) / stride + 1 即为 W 方向能切分的 patches 数量
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)  # tensor(1, H_pad, W_pad) 在原图右下角使用 0 填充
        image = image.unfold(dimension=1, size=patch_size, step=stride).unfold(  # tensor(1, num_H, W_pad, patch_size)
            2, patch_size, stride).permute(  # tensor(1, num_H, num_W, patch_size, patch_size)
            1, 2, 0, 3, 4)  # tensor(num_H, num_W, 1, patch_size, patch_size)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)  # tensor(n_patches, 1, patch_size, patch_size)

        # for sub2 in image:  
        #     image_list.append(sub2)
        # 原来的实现太慢了，改为下面的方式
        image_list.extend(list(image[:,0].chunk(image.shape[0], dim=0)))  # [tensor(1, patch_size, patch_size) , ...]
    return image_list


def save_imgs_pickle(imgs_list, path, type, name, verbose=True):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            if verbose:
                print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)  # tensor(N, H, W)---[0., 1.]
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    normalize = Normalize([mean], [std])  # output[channel] = (input[channel] - mean[channel]) / std[channel]
    for i in imgs_list:
        n = normalize(i)  # 每张图减去整个数据集的均值，再除以整个数据集的方差
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))  # 然后将这幅图像再映射至 [0., 1.] 范围内 （！！！但实际上原图像周围有一大圈黑色，这个对于数据的分布影响很大）
        normal_list.append(n)
    return normal_list  # [tensor(1, H, W) , ...]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path',
                        # default="datasets/DRIVE",
                        default="datasets/CHASEDB1",
                        type=str,
                        help='the path of dataset',
                        # required=True,
                        )
    parser.add_argument('-dn', '--dataset_name',
                        # default="DRIVE",
                        default="CHASEDB1",
                        type=str,
                        help='the name of dataset', choices=['DRIVE', 'CHASEDB1', 'STARE', 'CHUAC', 'DCA1'],
                        # required=True,
                        )
    parser.add_argument('-ps', '--patch_size', default=144,  # 48
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=36,  # 6
                        help='the stride of image partition')
    args = parser.parse_args()
    with open('config.yaml', encoding='utf-8') as file:
        CFG = safe_load(file)  # 得到CFG为字典类型

    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "training")
    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "test")
