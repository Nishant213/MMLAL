import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def parse_args(parser):
    '''
    Standard argument parser
    '''
    args = parser.parse_args()
    if args.config_file and os.path.exists(args.config_file):
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
#        print (data)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args

class MyCEFocalLoss(nn.Module):
    def __init__(self, num_classes, gamma = 0.0, use_polyloss = False):
        super(MyCEFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.use_polyloss = use_polyloss
        
    def forward(self, pred, target):

        ce_loss = F.cross_entropy(pred, target, reduction = "none", ignore_index = -1)
        p_t = torch.exp(-ce_loss)
        
        if self.gamma > 0 and self.use_polyloss:
            fl_loss = (1 - p_t) ** self.gamma * ce_loss
            loss = fl_loss + 1.0 * torch.pow((1 - p_t), 1.0 + self.gamma)
        elif self.gamma > 0:
            fl_loss = (1 - p_t) ** self.gamma * ce_loss
            loss = fl_loss
        elif self.use_polyloss:
            poly_loss = (1 - p_t) * 1.0 + ce_loss
            loss = poly_loss
        else:
            loss = ce_loss
            
        loss = loss.mean()
        
        return loss
    
def compute_reco_loss(rep, label, mask, prob, strong_threshold=0.97, temp=0.5, num_queries=256, num_negatives=256, clip_encodings = None):
    batch_size, num_feat, im_w_, im_h = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    valid_pixel = label * mask

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = prob[:, i, :, :]
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries
        
        if clip_encodings is not None:
            seg_proto_list.append(clip_encodings[i].unsqueeze(0))
        else:
            seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
            
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        
        if clip_encodings is not None:
            seg_proto = seg_proto.float().to(rep.device)
        # print(seg_proto.shape)
        
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        return reco_loss / valid_seg


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index

def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)
