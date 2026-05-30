# code modified from github "Cross-Domain Few-Shot Semantic Segmentation" (PATNet)
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as Image

class DatasetISIC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = split  
        self.shot = shot
        self.num = num
        self.fold = fold 
        self.base_path = datapath


        self.categories = ['1', '2', '3']


        self.train_classes, self.test_class = self.get_train_test_classes()
        

        self.img_metadata_classwise = self.build_img_metadata_classwise()
        
        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {
            'query_images': query_img.unsqueeze(0),
            'query_labels': query_mask.unsqueeze(0),
            'query_name': query_name,

            'support_images': support_imgs.unsqueeze(0),
            'support_fg_labels': support_masks.unsqueeze(0),
            'support_names': support_names,

            'selected_class': query_name
        }

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')
        query_name = os.path.join(ann_path, query_id) + '_segmentation.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(ann_path, sid) + '_segmentation.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):

        if self.split == 'train':

            class_id = idx % len(self.train_classes)
            class_sample = self.train_classes[class_id]
        else:

            class_sample = self.test_class

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        return query_name, support_names, class_sample

    def get_train_test_classes(self):

        if self.fold == 1:
            return ['2', '3'], '1'  
        elif self.fold == 2:
            return ['1', '3'], '2' 
        elif self.fold == 3:
            return ['1', '2'], '3'  
        else:
            raise ValueError("Fold should be 1, 2, or 3.")

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {cat: [] for cat in self.categories}

        for cat in self.categories:

            if (self.split == 'train' and cat in self.train_classes) or (self.split == 'test' and cat == self.test_class):
                img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat))])
                for img_path in img_paths:
                    if os.path.basename(img_path).split('.')[1] == 'jpg':
                        img_metadata_classwise[cat].append(img_path)
        return img_metadata_classwise
