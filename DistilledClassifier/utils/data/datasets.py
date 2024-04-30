from torchvision.datasets import CIFAR10, CIFAR100
from utils.data.tiny_imagenet.dataset import TinyImageNetDataset
from torchvision.transforms import Compose, RandomCrop, functional as tvtf 
from utils.data import degtransforms, degradedimagedata as deg_data
import torch
from utils.data.cutout import Cutout
import numpy as np
from PIL import Image

class DegCIFAR10Dataset(CIFAR10):
    def __init__(self, data_dir, train = True, train_init_transform = None, teacher_transform = None, 
                 student_transform = None, val_transform = None, download = False, deg_type = 'jpeg', 
                 deg_range = None, deg_list = None, is_to_tensor = True, is_target_to_tensor = True, 
                 deg_to_tensor = None, cutout_method = None, cutout_length = None, 
                 cutout_apply_clean = True, cutout_apply_deg = True, cutout_independent = False):
        super().__init__(data_dir, train, train_init_transform, download = download)
        self.train = train
        self.teacher_transform = teacher_transform
        self.student_transform = student_transform
        self.deg_type = deg_type
        self.deg_range = deg_range
        if self.deg_range is None:
            self.deg_range = deg_data.get_type_range(self.deg_type)
        self.deg_list = deg_list 
        self.is_to_tensor = is_to_tensor
        self.is_target_to_tensor = is_target_to_tensor
        self.deg_to_tensor = deg_to_tensor
        self.cutout_method = cutout_method
        self.cutout_length = cutout_length
        self.cutout_apply_clean = cutout_apply_clean
        self.cutout_apply_deg = cutout_apply_deg
        self.cutout_independent = cutout_independent
        if cutout_method == 'Cutout' and self.cutout_length is not None:
            self.cutout = Cutout(length = cutout_length)
        self.epoch = 0
        self.deg_transform = Compose([degtransforms.DegApplyWithLevel(self.deg_type, self.deg_range, self.deg_list)])

    def __getitem__(self, index):
        """
        degradation & tensor are applied.
        """
        clean_img, target = super().__getitem__(index)
        orig_clean_img = clean_img.copy()

        if self.train:
            if self.teacher_transform: 
                clean_img, _ = self.teacher_transform(clean_img)
            if self.student_transform: 
                clean_img, _ = self.student_transform(orig_clean_img)
                deg_img, deg_lev = self.deg_transform(clean_img) if self.deg_type == 'jpeg' \
                                        else self.deg_transform(np.asarray(clean_img))
            else:
                deg_img, deg_lev = self.deg_transform(clean_img) if self.deg_type == 'jpeg' \
                                        else self.deg_transform(np.asarray(clean_img))
        else:
            deg_img, deg_lev = self.deg_transform(clean_img) if self.deg_type == 'jpeg' \
                                    else self.deg_transform(np.asarray(clean_img))
        if self.deg_type != 'jpeg':
            deg_img = Image.fromarray(np.uint8(deg_img.clip(0, 255)))
        
        if self.train: 
            # self.transform does not have the RandomCrop in the training process
            # Applying RandomCrop
            clean_img = tvtf.pad(clean_img, 4, 0, "constant")
            deg_img = tvtf.pad(deg_img, 4, 0, "constant")
            i, j, h, w = RandomCrop.get_params(clean_img, output_size=(32, 32))
            clean_img = tvtf.crop(clean_img, i, j, h, w)
            deg_img = tvtf.crop(deg_img, i, j, h, w)

        if self.is_to_tensor:
            tensor_clean_img = self.deg_to_tensor(clean_img)
            tensor_deg_img = self.deg_to_tensor(deg_img)
            imgs = (tensor_clean_img, tensor_deg_img)
        else:
            imgs = (clean_img, deg_img)

        # Applying cutout
        if self.train and self.cutout_method is not None:
            clean_img, deg_img = imgs
            if self.cutout_method == 'Cutout':
                if self.cutout_independent:
                    clean_mask = self.cutout.get_mask(clean_img)
                    deg_mask = self.cutout.get_mask(deg_img)
                else:
                    clean_mask = self.cutout.get_mask(clean_img)
                    deg_mask = clean_mask
                if self.cutout_apply_clean:
                    clean_img = self.cutout(clean_img, clean_mask)
                if self.cutout_apply_deg:
                    deg_img = self.cutout(deg_img, deg_mask)

            imgs = (clean_img, deg_img)

        if self.is_target_to_tensor:
            deg_lev = degtransforms.normalize_level(self.deg_type, deg_lev)
            tensor_target = torch.tensor(target)
            tensor_deg_lev = deg_lev
            targets = (tensor_target, tensor_deg_lev)
        else:
            targets = (target, deg_lev)

        return imgs, targets


