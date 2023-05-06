import os
import os.path as osp

import torch
import clip

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

def get_clip(clip_enc):
    if clip_enc == 'vit':
        clip_pretrained, _ = clip.load("ViT-B/32", jit=False, device = torch.device("cuda"))
    elif clip_enc == 'rn50':
        clip_pretrained, _ = clip.load("RN50x16", jit=False, device = torch.device("cuda"))
    
    return clip_pretrained


model = get_clip('rn50')

dataset_in_q = 'camvid'
text_qs_file = osp.join('helpers', 'labels', '{}.txt'.format(dataset_in_q))

with open(text_qs_file, 'r') as f:
    text_qs_names = f.read().splitlines()

text_qs = clip.tokenize(text_qs_names)

with torch.no_grad():
    z = model.encode_text(text_qs.to('cuda'))
    
z = z.cpu()

torch.save(z, 'helpers/labels/camvid_rn50.pt')
# z = z / z.norm(dim=-1, keepdim=True)


# plt.scatter(X_embedded[:,0],X_embedded[:,1])
