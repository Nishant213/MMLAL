import os
import os.path as osp

from tqdm import tqdm
import torch
import torch.nn as nn
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
from helpers import BuildDataLoader, BuildDataset
from helpers import get_camvid_label, get_cityscapes_label

from core_models import SegmenthorSL
import pytorch_lightning as pl
import torchvision.transforms.functional as transforms_f

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

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
    
    test_dataset = BuildDataset(root='datasets',
             dataset='CityScapes_688', idx_list='test',
                                    crop_size=[688,688],
                                    scale_size=(1.0, 1.0),
                                    is_train = False,
                                    use_augmentation_weak=False,
                                    use_augmentation_strong=False,
                                    use_augmentation_cutout=False,
                                    apply_transform=False)
    test_dataset_length = len(test_dataset)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = 'cpu'
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    model.to(device)
    model.eval()
    
    prompts = ['a photo of a road','a photo of a sidewalk', 'a photo of a building','a photo of a wall', 'a photo of a fence', 'a photo of a pole',
               'a photo of a traffic light', 'a photo of a traffic sign', 'a photo of vegetation', 'a photo of terrain', 'a photo of sky', 
               'a photo of a pedestrian', 'a photo of a rider', 'a photo of a car', 'a photo of a truck', 'a photo of a bus', 'a photo of a choo-choo train', 
           'a photo of a motorcycle', 'a photo of a bicycle']
    # prompts = ['a photo of a road','a photo of a sidewalk', 'Identify different types of buildings in Cityscapes images such as skyscrapers, houses, and offices','a photo of a wall', 'a photo of a fence', 'a photo of a pole',
    #            'a photo of a traffic light', 'Identify and classify different types of traffic signs in Cityscapes images, such as stop signs, yield signs, and speed limit signs', 'a photo of vegetation', 'a photo of terrain', 'a photo of sky', 
    #            'a photo of a pedestrian', 'a photo of a rider', 'a photo of a car', 'a photo of a truck', 'a photo of a bus', 'a photo of a choo-choo train', 
    #        'a photo of a motorcycle', 'a photo of a bicycle']
    
    hist = 0
    
    for idx in tqdm(range(test_dataset_length)):
        batch = test_dataset[idx]
        
        img, labels = batch['image'], batch['segmap']
        
        # labels[labels==-1] = 99
        # labels = labels.to(torch.uint8)
        # segmap = SegmentationMapsOnImage(np.array(labels), img.size)
        # labels = segmap.resize([2048,1024], interpolation="nearest")
        # img = img.resize([2048,1024])
        # print(np.array(labels).shape)
        # labels = transforms_f.to_tensor(np.array(labels))
        # labels[labels==99] = -1

        # img, labels = img.to(device), labels.to(device)
        # img = img.squeeze()
        labels = labels.to(device)
        inputs = processor(text=prompts, images=[img] * len(prompts), padding="max_length", return_tensors="pt")
        inputs.to(device)
        # predict
        with torch.no_grad():
            outputs = model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        # print(type(img))
        # print(img.size)
        # resize the outputs
        preds = nn.functional.interpolate(
            outputs.logits.unsqueeze(1),
            size=(img.size[1], img.size[0]),
            mode="bilinear"
        )
        threshold = 0.1
        preds = F.softmax(preds.squeeze(), dim = 0)
        conf, inds = torch.max(preds.unsqueeze(dim=0), dim=1)
        
        # print("Preds:", preds.shape)
        # flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))
        # print("Flat1:", flat_preds.shape)
        # # Initialize a dummy "unlabeled" mask with the threshold
        # flat_preds_with_treshold = torch.full((preds.shape[0], flat_preds.shape[-1]), threshold)
        # print("Flat3:",flat_preds_with_treshold.shape)
        # flat_preds_with_treshold[:preds.shape[0],:] = flat_preds
        # print("Flat3:",flat_preds_with_treshold.shape)
        # # Get the top mask index for each pixel
        # inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))
        # print("Flat4:",inds.shape)
        # inds = inds.unsqueeze(dim=0)
        # print("Flat5:",inds.shape)
        # if isinstance(inds, tuple):
            # predictions = inds[0]
        # else:
            # predictions = inds
        # 
        # predictions = F.interpolate(predictions, size = labels.shape[1:], mode='bilinear', align_corners=True)
        
        # predictions = F.softmax(predictions, dim = 1)
        # confidences, predictions = torch.max(predictions, dim=1)

        _hist = fast_hist(pred=inds.flatten().cpu().numpy(),
                              gtruth=labels.flatten().cpu().numpy(),
                              num_classes=19)
        hist += _hist
        
    iu, _ = calculate_iou(hist)
    
    get_stats(hist, iu, dataset = args.dataset)
    
    torch.cuda.empty_cache()
