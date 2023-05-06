import os
import os.path as osp

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import argparse
import matplotlib.pylab as plt

import numpy as np
import random
from datetime import datetime
import copy
from tabulate import tabulate
from tqdm import tqdm

from networks import get_network
from helpers import BuildDataLoader
from helpers import get_camvid_label, get_cityscapes_label

from core_models import SegmenthorSL
import pytorch_lightning as pl
import torchvision.transforms.functional as transforms_f


def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    # stretch ground truth labels by num_classes
    #   class 0  -> 0
    #   class 1  -> 19
    #   class 18 -> 342
    #
    # TP at 0 + 0, 1 + 1, 2 + 2 ...
    #
    # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    hist = np.bincount(num_classes * gtruth[mask].astype(int) + pred[mask],
                       minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

# -----------------------------------------------------------------------------
# Ref: https://github.com/NVIDIA/semantic-segmentation
# -----------------------------------------------------------------------------

def calculate_iou(hist_data):
    # acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / (hist_data.sum(axis=1) + 1e-10)
    acc_cls = np.nanmean(acc_cls)
    divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - np.diag(hist_data)
    iou = np.diag(hist_data) / (divisor + 1e-10)
    return iou, acc_cls

# -----------------------------------------------------------------------------
# Ref: https://github.com/NVIDIA/semantic-segmentation
# (Slightly modified)
# -----------------------------------------------------------------------------

def get_stats(hist, iu, dataset):
    
    iu_FP = hist.sum(axis=1) - np.diag(hist)
    iu_FN = hist.sum(axis=0) - np.diag(hist)
    iu_TP = np.diag(hist)
    
    if dataset == 'camvid':
        id2cat = get_camvid_label()
    elif dataset == 'pascal':
        id2cat = get_pascal_label()
    else:
        id2cat = get_cityscapes_label()
        
    tabulate_data = []
    
    header = ['Id', 'label', 'iU']
    header.extend(['Precision', 'Recall'])
    
    for class_id in range(len(iu)):
        class_data = []
        class_data.append(class_id)
        class_name = "{}".format(id2cat[class_id]) if class_id in id2cat else ''
        class_data.append(class_name)
        class_data.append(iu[class_id] * 100)
    
        # total_pixels = hist.sum()
        class_data.append((iu_TP[class_id] / (iu_TP[class_id] + iu_FP[class_id])) * 100)
        class_data.append((iu_TP[class_id] / (iu_TP[class_id] + iu_FN[class_id])) * 100)
        tabulate_data.append(class_data)
        
    print_str = str(tabulate((tabulate_data), headers=header, floatfmt='1.2f'))
    print(print_str)
    print("mIoU = {:.2f}".format(np.nanmean(iu) * 100))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Supervised Segmentation with Perfect Labels')
    ### 1. Data Loading
    parser.add_argument('--dataset', default = 'cityscapes688')
    parser.add_argument('--train_ul_n', default = None)
    parser.add_argument('--pathtomodel', default = None, type=str)
    args = parser.parse_args()
    
    data_loader_main = BuildDataLoader(dataset=args.dataset)
    print("Defined dataloader for {}".format(args.dataset))
    
    dataset_info = data_loader_main.build_evaluation()
    
    test_loader = torch.utils.data.DataLoader(
                dataset=dataset_info['test'],
                batch_size=4,
                shuffle=False,
                num_workers=4
                )
    
    test_dataset = iter(test_loader)
    test_dataset_length = len(test_loader)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = 'cpu'
    
    working_ckpt = 'lightning_logs/version_4/checkpoints/last.ckpt'
    model = SegmenthorSL.load_from_checkpoint(working_ckpt)
    
    model.to(device)
    model.eval()
    
    hist = 0
    
    for idx in tqdm(range(test_dataset_length)):
        batch = test_dataset.next()
        
        image, labels = batch['image'], batch['segmap']
        with torch.no_grad():
            outputs = model(image.to(device))
        if isinstance(outputs, tuple):
            predictions = outputs[0]
        else:
            predictions = outputs
        # print(predictions.shape)
        predictions = F.interpolate(predictions, size = labels.shape[1:], mode='bilinear', align_corners=True)
        predictions = F.softmax(predictions, dim = 1)
        print(predictions)
        # print("Prediction3: ", predictions.shape)
        # s = 0
        # for i in range(19):
        #     s+=predictions[0][i][0][0]
        # print("SUM = ",s)
        confidences, predictions = torch.max(predictions, dim=1)
        # print(predictions.shape)
        # print(confidences.shape)
        _hist = fast_hist(pred=predictions.flatten().cpu().numpy(),
                              gtruth=labels.flatten().cpu().numpy(),
                              num_classes=data_loader_main.classes)
        hist += _hist
        
    iu, _ = calculate_iou(hist)
    
    get_stats(hist, iu, dataset = args.dataset)
    
    torch.cuda.empty_cache()
