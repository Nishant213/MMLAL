import os
import os.path as osp

from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transforms_f
import torch.utils.data.sampler as sampler

from tqdm import tqdm
from .dataset_transforms import do_transforms
from .dataset_colormaps import cityscapes_class_map

from catalyst.data.sampler import DistributedSamplerWrapper

class BuildDataset(Dataset):
    def __init__(self,
                 root,
                 dataset,
                 idx_list,
                 is_train=True,
                 pseudo_dir=None,
                 crop_size=(512, 512),
                 scale_size=(0.5, 2.0),
                 use_augmentation_weak=False,
                 use_augmentation_strong=False,
                 use_augmentation_cutout=False,
                 apply_transform=True):
        
        self.root = root
        self.dataset = dataset
        
        idx_list_filename = osp.join(self.root, self.dataset, 'ImageSets', idx_list + '.txt')
                
        with open(idx_list_filename, 'r') as txtfile:
                self.idx_list = txtfile.read().splitlines()
            
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.use_augmentation_weak = use_augmentation_weak
        self.use_augmentation_strong = use_augmentation_strong
        self.use_augmentation_cutout = use_augmentation_cutout
        self.apply_transform = apply_transform

        if is_train:
            self.folder_loc = 'train'
        else:
            self.folder_loc = 'val'
            
        self.pseudo_dir = pseudo_dir
        
    def __getitem__(self, index):
        if self.dataset == 'CamVid':
            
            image = Image.open(osp.join(self.root, self.dataset, 'images', '{}.png'.format(self.idx_list[index])))
            segmap = Image.open(osp.join(self.root, self.dataset, 'annots', '{}.png'.format(self.idx_list[index])))
            segmap = np.array(segmap)
            segmap = Image.fromarray(np.where(segmap == 11, 255, segmap))

        if self.dataset == 'CityScapes_688':
            
            image = Image.open(osp.join(self.root, self.dataset, 'images', self.folder_loc, '{}.png'.format(self.idx_list[index])))
            segmap = Image.open(osp.join(self.root, self.dataset, 'annots', self.folder_loc, '{}.png'.format(self.idx_list[index])))
            segmap = Image.fromarray(cityscapes_class_map(np.array(segmap)))
            
        if self.dataset == 'CityScapes_800':
            
            image = Image.open(osp.join(self.root, self.dataset, 'images', self.folder_loc, '{}.png'.format(self.idx_list[index])))
            
            segmap_filename = self.idx_list[index].replace("_leftImg8bit", "_gtFine_labelIds")
            segmap = Image.open(osp.join(self.root, self.dataset, 'annots', self.folder_loc, '{}.png'.format(segmap_filename)))
            segmap = Image.fromarray(cityscapes_class_map(np.array(segmap)))
            
        if self.pseudo_dir is not None:
          pseudo_label = Image.open(osp.join(self.pseudo_dir, '{}.png').format(self.idx_list[index]))
          
          pseudo_label_mask = osp.join(self.pseudo_dir, '{}_mask.png').format(self.idx_list[index])
          if os.path.isfile(pseudo_label_mask):
              pseudo_label_mask = Image.open(pseudo_label_mask)
          else:
              pseudo_label_mask = None
        else:
            pseudo_label = None
            pseudo_label_mask = None
            
        image_width, image_height = image.size
        
        if self.apply_transform:
            image, segmap, pseudo_label = do_transforms(image,
                                                        segmap,
                                                        pseudo_label=pseudo_label,
                                                        ratio_range=self.scale_size,
                                                        crop_size=self.crop_size,
                                                        use_augmentation_weak=self.use_augmentation_weak,
                                                        use_augmentation_strong=self.use_augmentation_strong,
                                                        use_augmentation_cutout=self.use_augmentation_cutout)
        
            image = transforms_f.to_tensor(image)
        segmap = (transforms_f.to_tensor(segmap) * 255).long()
        segmap[segmap == 255] = -1  # invalid pixels are re-mapped to index -1
        
        if self.apply_transform:
            image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if pseudo_label is None:
            pseudo_label = torch.Tensor([0])
        else:
            pseudo_label = (transforms_f.to_tensor(pseudo_label) * 255).long()
            pseudo_label[pseudo_label == 255] = -1
            
        if pseudo_label_mask is None:
            pseudo_label_mask = torch.Tensor([0])
        else:
            pseudo_label_mask = (transforms_f.to_tensor(pseudo_label_mask) * 255).long()
            
        return {'image': image,
                'segmap': segmap.squeeze(0),
                'pseudo_label': pseudo_label.squeeze(0),
                'pseudo_label_mask': pseudo_label_mask.squeeze(0),
                'name': self.idx_list[index]
                }
    
    def __len__(self):
        return len(self.idx_list)

class BuildDataLoader:
    def __init__(self, dataset, pseudo_dir = None):
        
        if dataset == 'camvid':
            self.root = 'datasets'
            self.dataset = 'CamVid'
            self.im_size = [360, 480]
            self.crop_size = [360, 480]
            self.scale_size = (0.5, 2.0)
            self.classes = 11
            
            with open(osp.join('helpers','labels','camvid.txt')) as f:
                self.label_list = f.read().splitlines()
                
        elif dataset == 'cityscapes688':
            self.root = 'datasets'
            self.dataset = 'CityScapes_688'
            self.im_size = [688, 688]
            self.crop_size = [688, 688]
            self.scale_size = (0.5, 2.0)
            self.classes = 19
            
            with open(osp.join('helpers','labels','cityscapes.txt')) as f:
                self.label_list = f.read().splitlines()
            
        elif dataset == 'cityscapes800':
            self.root = 'datasets'
            self.dataset = 'CityScapes_800'
            self.im_size = [1024, 2048]
            self.crop_size = [768, 768]
            self.scale_size = (0.5, 2.0)
            self.classes = 19
            
            with open(osp.join('helpers','labels','cityscapes.txt')) as f:
                self.label_list = f.read().splitlines()
            
        self.pseudo_dir = pseudo_dir

    def build_supervised(self, args):
        
        train_l_dataset = BuildDataset(root=self.root,
                                       dataset=self.dataset,
                                       idx_list=args.train_l_n,
                                       crop_size=self.crop_size,
                                       scale_size=self.scale_size,
                                       is_train = True,
                                       use_augmentation_weak=True,
                                       use_augmentation_strong=False,
                                       use_augmentation_cutout=False,
                                       )
        
        if args.train_ul_n is not None:
            train_ul_dataset = BuildDataset(root=self.root,
                                           dataset=self.dataset,
                                           idx_list=args.train_ul_n,
                                           pseudo_dir=self.pseudo_dir,
                                           crop_size=self.crop_size,
                                           scale_size=self.scale_size,
                                           is_train = True,
                                           use_augmentation_weak=True,
                                           use_augmentation_strong=True,
                                           use_augmentation_cutout=True,
                                           )
            
            train_ul_label_dataset = BuildDataset(root=self.root,
                                                  dataset=self.dataset,
                                                  idx_list=args.train_ul_n,
                                                  pseudo_dir=self.pseudo_dir,
                                                  crop_size=self.im_size,
                                                  scale_size=(1.0, 1.0),
                                                  is_train = True,
                                                  use_augmentation_weak=False,
                                                  use_augmentation_strong=False,
                                                  use_augmentation_cutout=False,
                                                  )
            
        else:
            train_ul_dataset = None
            train_ul_label_dataset = None
        
        test_dataset = BuildDataset(root=self.root,
                                    dataset=self.dataset,
                                    idx_list='test',
                                    crop_size=self.im_size,
                                    scale_size=(1.0, 1.0),
                                    is_train = False,
                                    use_augmentation_weak=False,
                                    use_augmentation_strong=False,
                                    use_augmentation_cutout=False,
                                    )
        
        num_samples = args.batch_size * args.batch_iters
        
        return {'train_l': train_l_dataset,
                'train_ul': train_ul_dataset,
                'train_ul_4label': train_ul_label_dataset,
                'test': test_dataset,
                'nc': self.classes,
                'ns': num_samples,
                'label_list': self.label_list,
                }
    
    def build_evaluation(self, apply_transform = True):
        
        test_dataset = BuildDataset(root=self.root,
                                    dataset=self.dataset,
                                    idx_list='test',
                                    pseudo_dir=self.pseudo_dir,
                                    crop_size=self.im_size,
                                    scale_size=(1.0, 1.0),
                                    is_train = False,
                                    use_augmentation_weak=False,
                                    use_augmentation_strong=False,
                                    use_augmentation_cutout=False,
                                    apply_transform=apply_transform)
        
        
        return {'test': test_dataset,
                'label_list': self.label_list,
                }
    
# if __name__ == "__main__":
    
#     try_dataset = BuildDataset(root='datasets',
#                                dataset='CamVid',
#                                idx_list = 'train',
#                                is_train=True,
#                                crop_size=(360,480),
#                                scale_size=(1.0, 1.0))
    
#     #Sample a random entry
#     z = try_dataset[random.randint(0, len(try_dataset))]
    
