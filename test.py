import argparse
import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import test
from utils import losses
from utils.helpers import get_instance


def main(data_path, weight_path, CFG, show):
    checkpoint = torch.load(weight_path)
    CFG_ck = checkpoint['config']
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False,  num_workers=16, pin_memory=True)
    model = get_instance(models, 'model', CFG_ck)
    loss = get_instance(losses, 'loss', CFG_ck)
    test(model, loss, CFG, checkpoint, test_loader, data_path, show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/lwt/data_pro/vessel/DCA1", type=str,
                        help="the path of dataset",required=True)
    parser.add_argument("-wp", "--wetght_path", default="pretrain_weight/DCA1/checkpoint-epoch40.pth", type=str,
                        help='the path of wetght.pt',required=True)
    parser.add_argument("--show", help="save predict image",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()
    with open("config.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
    main(args.dataset_path, args.wetght_path, CFG, args.show)
