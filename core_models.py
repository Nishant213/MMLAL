import os
import os.path as osp

import pytorch_lightning as pl

from networks import get_network
import torch.optim as optim
from helpers import MyCEFocalLoss, SegmentationMetric
from helpers import compute_reco_loss, label_onehot, negative_index_sampler

import torch.nn.functional as F

from catalyst.data.sampler import DistributedSamplerWrapper
import torch.utils.data.sampler as sampler
import torch

from PIL import Image
import numpy as np

from helpers import create_camvid_label_colormap, create_cityscapes_label_colormap, get_colored_output

class SegmenthorSL(pl.LightningModule):
    def __init__(self, 
                 args,
                 dataset_info
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = 0.01 #dummy for PL

        self.net = get_network(backbone=args.backbone,
                               rep_size=args.rep_size,
                               num_classes=dataset_info['nc']
                               )
        
        if args.clip_enc is not None:
            self.label_encodings = torch.load(osp.join('helpers',
                                                       'labels',
                                                       '{}_{}.pt'.format(args.dataset, args.clip_enc)
                                                       )
                                              )
        else:
            self.label_encodings = None
        
        self.val_iou = SegmentationMetric(dataset_info['nc'])
        
        self.loss_fx = MyCEFocalLoss(num_classes = dataset_info['nc'],
                                     gamma = args.loss_fx_focal_weight,
                                     use_polyloss = args.use_polyloss
                                     )

        pl.seed_everything(args.fixed_seed)
        
    def forward(self, img):
        
        prediction = self.net(img)
        
        return prediction
    
    def training_step(self, batch, batch_idx):
        
        imgs, labels = batch['labeled']['image'], batch['labeled']['segmap']
        
        outputs = self(imgs)
        
        if isinstance(outputs, tuple):
            predictions = outputs[0]
            representations = outputs[1]
            
            with torch.no_grad():
                mask = F.interpolate((labels.unsqueeze(1) >= 0).float(), size = predictions.shape[2:], mode='nearest')
                label = F.interpolate(label_onehot(labels, num_segments = self.hparams["dataset_info"]['nc']),
                                      size = predictions.shape[2:],
                                      mode='nearest'
                                      )
                
                prob = torch.softmax(predictions, dim = 1)
                
            loss_reco = compute_reco_loss(representations,
                                          label,
                                          mask,
                                          prob,
                                          strong_threshold=self.hparams['args'].reco_str_th,
                                          temp=self.hparams['args'].reco_temp,
                                          num_queries=self.hparams['args'].reco_num_queries,
                                          num_negatives=self.hparams['args'].reco_num_negatives,
                                          clip_encodings=self.label_encodings,
                                          )
        else:
            predictions = outputs
            loss_reco = 0.0
            
        predictions = F.interpolate(predictions, size = labels.shape[1:], mode='bilinear', align_corners=True)
        loss_cls = self.loss_fx(predictions, labels)
        
        loss = loss_cls + loss_reco
        
        # ## Test for unused parameters in pytorch
        # for name, param in self.net.named_parameters():
        #    if param.grad is None:
        #        print(name)
        
        self.log("train_loss_classifier", loss_cls)
        self.log("train_loss_reco", loss_reco)
        self.log("train_loss", loss)
        
        return loss
        
    def train_dataloader(self):
        
        if self.hparams['args'].gpus > 1:
            sampler_x = DistributedSamplerWrapper(
                sampler = sampler.RandomSampler(data_source=self.hparams["dataset_info"]['train_l'],
                                                replacement=True,
                                                num_samples=self.hparams["dataset_info"]['ns'])
                )
        else:
            sampler_x = sampler.RandomSampler(data_source=self.hparams["dataset_info"]['train_l'],
                                              replacement=True,
                                              num_samples=self.hparams["dataset_info"]['ns'])
            
        train_labeled_loader =  torch.utils.data.DataLoader(dataset = self.hparams["dataset_info"]['train_l'],
                                                            sampler = sampler_x,
                                                            batch_size = self.hparams["args"].batch_size,
                                                            drop_last = True,
                                                            num_workers = 4,
                                                            pin_memory=True
                                                            )
        
        return {"labeled": train_labeled_loader}
    
    def label_dataloader(self):
        
        if self.hparams['args'].gpus > 1:
            sampler_x = torch.utils.data.distributed.DistributedSampler(self.hparams["dataset_info"]['train_ul_4label'])
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['train_ul_4label'],
                sampler = sampler_x,
                batch_size = 1,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        else:
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['train_ul_4label'],
                batch_size = 1,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
    
    def val_dataloader(self):
        
        if self.hparams['args'].gpus > 1:
            sampler_x = torch.utils.data.distributed.DistributedSampler(self.hparams["dataset_info"]['test'])
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['test'],
                sampler = sampler_x,
                batch_size = self.hparams["args"].batch_size,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        else:
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['test'],
                batch_size = self.hparams["args"].batch_size,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        
    def validation_step(self, batch, batch_idx):
        
        imgs, labels = batch['image'], batch['segmap']
        
        predictions = self(imgs)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = F.interpolate(predictions, size = labels.shape[1:], mode='bilinear', align_corners=True)
        self.val_iou.update(labels, predictions)
        
    def predict_step(self, batch, batch_idx):
        
        imgs, labels, img_names, img_heights, img_widths = batch['image'],\
            batch['segmap'],\
                batch["name"],\
                    batch["image_height"],\
                        batch["image_width"]
        pl_mask = batch['pseudo_label_mask']
        
        pseudo_labels_logits_2x = self(F.interpolate(imgs,
                                                     scale_factor=2,
                                                     mode = 'bilinear',
                                                     align_corners=True
                                                     )
                                       )
        
        pseudo_labels_logits_2x = F.interpolate(
            pseudo_labels_logits_2x,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=True,
            )
        
        pseudo_labels_logits = self(imgs)
        
        pseudo_labels_logits_halfx = self(F.interpolate(imgs,
                                                        scale_factor=0.5,
                                                        mode = 'bilinear',
                                                        align_corners=True
                                                        )
                                          )
        
        pseudo_labels_logits = F.interpolate(
            pseudo_labels_logits,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=True,
            )
        
        pseudo_labels_logits_halfx = F.interpolate(
            pseudo_labels_logits_halfx,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=True,
            )

        pseudo_labels_logits = (
            pseudo_labels_logits
            + pseudo_labels_logits_halfx
            + pseudo_labels_logits_2x
            ) / 3
        
        pseudo_labels_logits = F.softmax(pseudo_labels_logits, dim=1)
        confidences_pl, predictions_pl = torch.max(pseudo_labels_logits, dim=1)
        
        predictions_pl[confidences_pl < self.hparams["args"].pixel_threshold] = 255
        
        if self.hparams["args"].label_grid_counts > 0:

            if len(pl_mask.shape) == 1:
                pl_mask = torch.zeros(labels.shape).to(self.device)
            
            if self.hparams["args"].dataset == 'camvid':
                adaptive_pool_size = (int(360/self.hparams["args"].label_grid_size), int(480/self.hparams["args"].label_grid_size))
            else:
                adaptive_pool_size = (int(688/self.hparams["args"].label_grid_size), int(688/self.hparams["args"].label_grid_size))
            
            pred_ent = (-pseudo_labels_logits * torch.log2(pseudo_labels_logits)).sum(dim=1)
            pred_ent[pl_mask > 0] = 0
            
            pred_avg = F.adaptive_avg_pool2d(pred_ent, output_size = adaptive_pool_size)
            pred_avg[:,0,:] = pred_avg[:,-1,:] = pred_avg[:,:,-1] = pred_avg[:,:,0] = 0 #littlehack

            for sub_idx in range(len(pred_avg)):

                v, k = torch.topk(pred_avg[sub_idx].flatten(), self.hparams["args"].label_grid_counts)
                indices = np.array(
                    np.unravel_index(k.cpu().numpy(), pred_avg[sub_idx].shape)
                    ).T

                for len_indices in range(len(indices)):
                    x1 = indices[len_indices, 1] * self.hparams["args"].label_grid_size
                    x2 = indices[len_indices, 1] * self.hparams["args"].label_grid_size + self.hparams["args"].label_grid_size
                    y1 = indices[len_indices, 0] * self.hparams["args"].label_grid_size
                    y2 = indices[len_indices, 0] * self.hparams["args"].label_grid_size + self.hparams["args"].label_grid_size
                    pl_mask[sub_idx, y1:y2, x1:x2] = 1

            pl_mask[pl_mask > 0] = 255

            predictions_pl[pl_mask > 0] = labels[pl_mask > 0]
            predictions_pl[predictions_pl == -1] = 255
        
        for idx_sub in range(len(predictions_pl)):
            
            temp_pseudo = Image.fromarray(
                predictions_pl[idx_sub].cpu().numpy().astype(np.uint8)
            )
                        
            temp_pseudo.save(
                osp.join(
                    "{}".format(self.hparams["args"].pseudo_dir),
                    "{}.png".format(img_names[idx_sub]),
                )
            )
            
            if self.hparams["args"].label_grid_counts > 0:
                
                temp_pseudo_mask = Image.fromarray(
                    pl_mask[idx_sub].cpu().numpy().astype(np.uint8)
                )

                temp_pseudo_mask.save(
                    osp.join(
                        "{}".format(self.hparams["args"].pseudo_dir),
                        "{}_mask.png".format(img_names[idx_sub]),
                    )
                )
        
    def validation_epoch_end(self, validation_step_outputs):
        
        pixAcc, iou = self.val_iou.get()
        self.log("pix_acc_epoch", pixAcc)
        self.log("val_iou_epoch", iou, prog_bar = True)
    
        self.val_iou.reset()
    
    def configure_optimizers(self):
            
        params_list = [
            {"params": self.net.backbone.parameters(), "lr": self.hparams['args'].start_lr},
            ]
        if hasattr(self.net, "aspp_dl"):
            params_list.append(
                {"params": self.net.aspp_dl.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "low_level_1x1"):
            params_list.append(
                {"params": self.net.low_level_1x1.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "head_classifier"):
            params_list.append(
                {"params": self.net.head_classifier.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "head_representation"):
            params_list.append(
                {"params": self.net.head_representation.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "scratch"):
            params_list.append(
                {"params": self.net.scratch.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        
        optimizer = optim.SGD(params_list,
                              lr=self.hparams['args'].start_lr,
                              weight_decay=self.hparams['args'].weight_decay,
                              momentum=0.9,
                              nesterov=True
                              )
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda x: pow(1.0 - x / self.hparams['args'].max_epochs, 0.9)
                                                )
        
        return ([optimizer], [scheduler])
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SegmenthorSL")
        
        parser.add_argument("--exp_name", type = str, default = None)
        
        parser.add_argument("--backbone", type = str, default = 'mbv2_deeplab')
        parser.add_argument("--clip_enc", type = str, default = None)
        
        parser.add_argument("--rep_size", type = int, default = 0)
        parser.add_argument("--reco_str_th", type = float, default = 0.95)
        parser.add_argument("--reco_temp", type = float, default = 0.5)
        parser.add_argument("--reco_num_queries", type = int, default = 128)
        parser.add_argument("--reco_num_negatives", type = int, default = 256)
        
        parser.add_argument('--start_lr', type = float, default = 1e-2)
        parser.add_argument('--weight_decay', type = float, default = 1e-4)
        
        parser.add_argument('--loss_fx_focal_weight', type = float, default = 0)
        parser.add_argument('--use_polyloss', action = 'store_true', default = False)
        
        parser.add_argument('--fixed_seed', type = int, default = 3108)
        
        return parent_parser

class SegmenthorSSL(pl.LightningModule):
    def __init__(self, 
                 args,
                 dataset_info
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = 0.01 #dummy for PL

        self.net = get_network(backbone=args.backbone,
                               rep_size=args.rep_size,
                               num_classes=dataset_info['nc']
                               )
        
        if args.clip_enc is not None:
            self.label_encodings = torch.load(osp.join('helpers',
                                                       'labels',
                                                       '{}_{}.pt'.format(args.dataset, args.clip_enc)
                                                       )
                                              )
        else:
            self.label_encodings = None
            
        self.val_iou = SegmentationMetric(dataset_info['nc'])
        
        self.loss_fx = MyCEFocalLoss(num_classes = dataset_info['nc'],
                                      gamma = args.loss_fx_focal_weight,
                                      use_polyloss = args.use_polyloss
                                      )

        pl.seed_everything(args.fixed_seed)
        
    def forward(self, img):
        
        prediction = self.net(img)
        
        return prediction
    
    def training_step(self, batch, batch_idx):
        
        imgs = torch.cat((batch['labeled']['image'], batch['unlabeled']['image']))
        labels = torch.cat((batch['labeled']['segmap'], batch['unlabeled']['pseudo_label']))
        
        outputs = self(imgs)
        
        if isinstance(outputs, tuple):
            predictions = outputs[0]
            representations = outputs[1]
            
            with torch.no_grad():
                mask = F.interpolate((labels.unsqueeze(1) >= 0).float(), size = predictions.shape[2:], mode='nearest')
                label = F.interpolate(label_onehot(labels, num_segments = self.hparams["dataset_info"]['nc']),
                                      size = predictions.shape[2:],
                                      mode='nearest'
                                      )
                
                prob = torch.softmax(predictions, dim = 1)
                
            loss_reco = compute_reco_loss(representations,
                                          label,
                                          mask,
                                          prob,
                                          strong_threshold=self.hparams['args'].reco_str_th,
                                          temp=self.hparams['args'].reco_temp,
                                          num_queries=self.hparams['args'].reco_num_queries,
                                          num_negatives=self.hparams['args'].reco_num_negatives,
                                          clip_encodings=self.label_encodings,
                                          )
        else:
            predictions = outputs
            loss_reco = 0.0
            
        predictions = F.interpolate(predictions, size = labels.shape[1:], mode='bilinear', align_corners=True)
        loss_cls = self.loss_fx(predictions, labels)
        
        loss = loss_cls + loss_reco
        
        self.log("train_loss_classifier", loss_cls)
        self.log("train_loss_reco", loss_reco)
        self.log("train_loss", loss)
        
        return loss
        
    def train_dataloader(self):
        
        if self.hparams['args'].gpus > 1:
            sampler_x = DistributedSamplerWrapper(
                sampler = sampler.RandomSampler(data_source=self.hparams["dataset_info"]['train_l'],
                                                replacement=True,
                                                num_samples=self.hparams["dataset_info"]['ns'])
                )
            sampler_y = DistributedSamplerWrapper(
                sampler = sampler.RandomSampler(data_source=self.hparams["dataset_info"]['train_ul'],
                                                replacement=True,
                                                num_samples=self.hparams["dataset_info"]['ns']*2)
                )
        else:
            sampler_x = sampler.RandomSampler(data_source=self.hparams["dataset_info"]['train_l'],
                                              replacement=True,
                                              num_samples=self.hparams["dataset_info"]['ns'])
            sampler_y = sampler.RandomSampler(data_source=self.hparams["dataset_info"]['train_ul'],
                                              replacement=True,
                                              num_samples=self.hparams["dataset_info"]['ns']*2)
            
        train_labeled_loader =  torch.utils.data.DataLoader(dataset = self.hparams["dataset_info"]['train_l'],
                                                            sampler = sampler_x,
                                                            batch_size = self.hparams["args"].batch_size,
                                                            drop_last = True,
                                                            num_workers = 4,
                                                            pin_memory=True
                                                            )
        train_ulabeled_loader = torch.utils.data.DataLoader(dataset = self.hparams["dataset_info"]['train_ul'],
                                                            sampler = sampler_y,
                                                            batch_size = self.hparams["args"].batch_size*2,
                                                            drop_last = True,
                                                            num_workers = 4,
                                                            pin_memory=True
                                                            )
        
        return {"labeled": train_labeled_loader, "unlabeled": train_ulabeled_loader}
    
    def label_dataloader(self):
        
        if self.hparams['args'].gpus > 1:
            sampler_x = torch.utils.data.distributed.DistributedSampler(self.hparams["dataset_info"]['train_ul_4label'])
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['train_ul_4label'],
                sampler = sampler_x,
                batch_size = 1,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        else:
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['train_ul_4label'],
                batch_size = 1,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
    
    def val_dataloader(self):
        
        if self.hparams['args'].gpus > 1:
            sampler_x = torch.utils.data.distributed.DistributedSampler(self.hparams["dataset_info"]['test'])
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['test'],
                sampler = sampler_x,
                batch_size = self.hparams["args"].batch_size,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        else:
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['test'],
                batch_size = self.hparams["args"].batch_size,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        
    def validation_step(self, batch, batch_idx):
        
        imgs, labels = batch['image'], batch['segmap']
        
        predictions = self(imgs)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
            
        predictions = F.interpolate(predictions, size = labels.shape[1:], mode='bilinear', align_corners=True)
        self.val_iou.update(labels, predictions)
        
    def predict_step(self, batch, batch_idx):
        
        imgs, labels, img_names = batch['image'], batch['segmap'], batch["name"]
        pl_mask = batch['pseudo_label_mask']
        
        pseudo_labels_logits_2x = self(F.interpolate(imgs,
                                                     scale_factor=2,
                                                     mode = 'bilinear',
                                                     align_corners=True
                                                     )
                                       )
        
        pseudo_labels_logits_2x = F.interpolate(
            pseudo_labels_logits_2x,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=True,
            )
        
        pseudo_labels_logits = self(imgs)
        
        pseudo_labels_logits_halfx = self(F.interpolate(imgs,
                                                        scale_factor=0.5,
                                                        mode = 'bilinear',
                                                        align_corners=True
                                                        )
                                          )
        
        pseudo_labels_logits = F.interpolate(
            pseudo_labels_logits,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=True,
            )
        
        pseudo_labels_logits_halfx = F.interpolate(
            pseudo_labels_logits_halfx,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=True,
            )

        pseudo_labels_logits = (
            pseudo_labels_logits
            + pseudo_labels_logits_halfx
            + pseudo_labels_logits_2x
            ) / 3
        
        pseudo_labels_logits = F.softmax(pseudo_labels_logits, dim=1)
        confidences_pl, predictions_pl = torch.max(pseudo_labels_logits, dim=1)
        
        predictions_pl[confidences_pl < self.hparams["args"].pixel_threshold] = 255
        
        if self.hparams["args"].label_grid_counts > 0:

            if len(pl_mask.shape) == 1:
                pl_mask = torch.zeros(labels.shape).to(self.device)
            
            if self.hparams["args"].dataset == 'camvid':
                adaptive_pool_size = (int(360/self.hparams["args"].label_grid_size), int(480/self.hparams["args"].label_grid_size))
            else:
                adaptive_pool_size = (int(688/self.hparams["args"].label_grid_size), int(688/self.hparams["args"].label_grid_size))
            
            pred_ent = (-pseudo_labels_logits * torch.log2(pseudo_labels_logits)).sum(dim=1)
            pred_ent[pl_mask > 0] = 0
            
            pred_avg = F.adaptive_avg_pool2d(pred_ent, output_size = adaptive_pool_size)
            pred_avg[:,0,:] = pred_avg[:,-1,:] = pred_avg[:,:,-1] = pred_avg[:,:,0] = 0 #littlehack

            for sub_idx in range(len(pred_avg)):

                v, k = torch.topk(pred_avg[sub_idx].flatten(), self.hparams["args"].label_grid_counts)
                indices = np.array(
                    np.unravel_index(k.cpu().numpy(), pred_avg[sub_idx].shape)
                    ).T

                for len_indices in range(len(indices)):
                    x1 = indices[len_indices, 1] * self.hparams["args"].label_grid_size
                    x2 = indices[len_indices, 1] * self.hparams["args"].label_grid_size + self.hparams["args"].label_grid_size
                    y1 = indices[len_indices, 0] * self.hparams["args"].label_grid_size
                    y2 = indices[len_indices, 0] * self.hparams["args"].label_grid_size + self.hparams["args"].label_grid_size
                    pl_mask[sub_idx, y1:y2, x1:x2] = 1

            pl_mask[pl_mask > 0] = 255

            predictions_pl[pl_mask > 0] = labels[pl_mask > 0]
            predictions_pl[predictions_pl == -1] = 255
        
        for idx_sub in range(len(predictions_pl)):
            
            temp_pseudo = Image.fromarray(
                predictions_pl[idx_sub].cpu().numpy().astype(np.uint8)
            )
                        
            temp_pseudo.save(
                osp.join(
                    "{}".format(self.hparams["args"].pseudo_dir),
                    "{}.png".format(img_names[idx_sub]),
                )
            )
            
            if self.hparams["args"].label_grid_counts > 0:
                
                temp_pseudo_mask = Image.fromarray(
                    pl_mask[idx_sub].cpu().numpy().astype(np.uint8)
                )

                temp_pseudo_mask.save(
                    osp.join(
                        "{}".format(self.hparams["args"].pseudo_dir),
                        "{}_mask.png".format(img_names[idx_sub]),
                    )
                )
        
    def validation_epoch_end(self, validation_step_outputs):
        
        pixAcc, iou = self.val_iou.get()
        self.log("pix_acc_epoch", pixAcc)
        self.log("val_iou_epoch", iou, prog_bar = True)
    
        self.val_iou.reset()
    
    def configure_optimizers(self):
            
        params_list = [
            {"params": self.net.backbone.parameters(), "lr": self.hparams['args'].start_lr},
            ]
        if hasattr(self.net, "aspp_dl"):
            params_list.append(
                {"params": self.net.aspp_dl.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "low_level_1x1"):
            params_list.append(
                {"params": self.net.low_level_1x1.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "head_classifier"):
            params_list.append(
                {"params": self.net.head_classifier.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "head_representation"):
            params_list.append(
                {"params": self.net.head_representation.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        if hasattr(self.net, "scratch"):
            params_list.append(
                {"params": self.net.scratch.parameters(), "lr": self.hparams['args'].start_lr * 10}
                )
        
        optimizer = optim.SGD(params_list,
                              lr=self.hparams['args'].start_lr,
                              weight_decay=self.hparams['args'].weight_decay,
                              momentum=0.9,
                              nesterov=True
                              )
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda x: pow(1.0 - x / self.hparams['args'].max_epochs, 0.9)
                                                )
        
        return ([optimizer], [scheduler])
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SegmenthorSL")
        
        parser.add_argument("--exp_name", type = str, default = None)
        
        parser.add_argument("--backbone", type = str, default = 'mbv2_deeplab')
        parser.add_argument("--clip_enc", type = str, default = None)
        
        parser.add_argument("--rep_size", type = int, default = 0)
        parser.add_argument("--reco_str_th", type = float, default = 0.95)
        parser.add_argument("--reco_temp", type = float, default = 0.5)
        parser.add_argument("--reco_num_queries", type = int, default = 128)
        parser.add_argument("--reco_num_negatives", type = int, default = 256)
        
        parser.add_argument('--start_lr', type = float, default = 1e-2)
        parser.add_argument('--weight_decay', type = float, default = 1e-4)
        
        parser.add_argument('--loss_fx_focal_weight', type = float, default = 0)
        parser.add_argument('--use_polyloss', action = 'store_true', default = False)
        
        parser.add_argument('--fixed_seed', type = int, default = 3108)
        
        return parent_parser

class SegmenthorViz(pl.LightningModule):
    def __init__(self, 
                  args,
                  dataset_info
                  ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = 0.01 #dummy for PL

        self.net = get_network(backbone=args.backbone,
                                rep_size=args.rep_size,
                                num_classes=dataset_info['nc']
                                )
        
        if args.clip_enc is not None:
            self.label_encodings = torch.load(osp.join('helpers',
                                                        'labels',
                                                        '{}_{}.pt'.format(args.dataset, args.clip_enc)
                                                        )
                                              )
        else:
            self.label_encodings = None
        
        self.val_iou = SegmentationMetric(dataset_info['nc'])
        
        self.loss_fx = MyCEFocalLoss(num_classes = dataset_info['nc'],
                                      gamma = args.loss_fx_focal_weight,
                                      use_polyloss = args.use_polyloss
                                      )

        pl.seed_everything(args.fixed_seed)
        
    def forward(self, img):
        
        prediction = self.net(img)
        
        return prediction
    
    def val_dataloader(self):
        
        if self.hparams['args'].gpus > 1:
            sampler_x = torch.utils.data.distributed.DistributedSampler(self.hparams["dataset_info"]['test'])
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['test'],
                sampler = sampler_x,
                batch_size = self.hparams["args"].batch_size,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        else:
            return torch.utils.data.DataLoader(
                dataset = self.hparams["dataset_info"]['test'],
                batch_size = self.hparams["args"].batch_size,
                shuffle = False,
                num_workers = 4,
                pin_memory=True
                )
        
    def validation_step(self, batch, batch_idx):
        
        imgs, labels = batch['image'], batch['segmap']
        
        predictions = self(imgs)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = F.interpolate(predictions, size = labels.shape[1:], mode='bilinear', align_corners=True)
        self.val_iou.update(labels, predictions)
        
    def predict_step(self, batch, batch_idx):
        
        imgs, labels, img_names = batch['image'], batch['segmap'], batch["name"]
        
        pseudo_labels_logits = self(imgs)
        
        pseudo_labels_logits = F.interpolate(
            pseudo_labels_logits,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=True,
            )
        
        pseudo_labels_logits = F.softmax(pseudo_labels_logits, dim=1)
        _, predictions_pl = torch.max(pseudo_labels_logits, dim=1)
        
        predictions_pl[labels == -1] = -1
        
        for idx_sub in range(len(predictions_pl)):
            
            if self.hparams["args"].dataset == 'camvid':
                colormap = create_camvid_label_colormap()
            else:
                colormap = create_cityscapes_label_colormap()
                
            temp_pseudo = Image.fromarray(
                get_colored_output(predictions_pl[idx_sub].cpu().numpy().astype(np.uint8), colormap)
            )
                        
            temp_pseudo.save(
                osp.join(
                    "{}".format(self.hparams["args"].pseudo_dir),
                    "{}.png".format(img_names[idx_sub]),
                )
            )
            
    def validation_epoch_end(self, validation_step_outputs):
        
        pixAcc, iou = self.val_iou.get()
        self.log("pix_acc_epoch", pixAcc)
        self.log("val_iou_epoch", iou, prog_bar = True)
    
        self.val_iou.reset()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SegmenthorSL")
        
        parser.add_argument("--exp_name", type = str, default = None)
        
        parser.add_argument("--backbone", type = str, default = 'mbv2_deeplab')
        parser.add_argument("--clip_enc", type = str, default = None)
        
        parser.add_argument("--rep_size", type = int, default = 0)
        parser.add_argument("--reco_str_th", type = float, default = 0.95)
        parser.add_argument("--reco_temp", type = float, default = 0.5)
        parser.add_argument("--reco_num_queries", type = int, default = 128)
        parser.add_argument("--reco_num_negatives", type = int, default = 256)
        
        parser.add_argument('--start_lr', type = float, default = 1e-2)
        parser.add_argument('--weight_decay', type = float, default = 1e-4)
        
        parser.add_argument('--loss_fx_focal_weight', type = float, default = 0)
        parser.add_argument('--use_polyloss', action = 'store_true', default = False)
        
        parser.add_argument('--fixed_seed', type = int, default = 3108)
        
        return parent_parser
