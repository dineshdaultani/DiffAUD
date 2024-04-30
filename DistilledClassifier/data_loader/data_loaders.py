from torchvision import transforms
from base import BaseDataLoader
import argparse
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
from utils.data.datasets import DegCIFAR10Dataset
from utils.data.image_datasets import load_data
from utils.data import degtransforms, degradedimagedata as deg_data

class CIFAR10DataLoader(BaseDataLoader):
    """
    Revised CIFAR10 data loader based on imagenet data loader
    """
    def __init__(self, data_dir, batch_size, data_val_dir = None, image_size = 32, train=True, class_cond = True, 
                 shuffle=True, validation_split=0.0, num_workers=1, deg_type = 'jpeg', deg_flag = 'deg', 
                 deg_types = None, severity = None):
        if train:
            transforms_ = transforms.Compose([transforms.RandomCrop(image_size, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(125.307/255.0, 122.961/255.0, 113.8575/255.0), 
                                                                   std=(51.5865/255.0, 50.847/255.0, 51.255/255.0)),
                                              ])
        else:
            transforms_ = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=(125.307/255.0, 122.961/255.0, 113.8575/255.0), 
                                                                   std=(51.5865/255.0, 50.847/255.0, 51.255/255.0)),
                                             ])
            
        if train or data_val_dir is None:
            self.dataset = load_data(
                                data_dir=data_dir,
                                batch_size=batch_size,
                                image_size=image_size,
                                transforms_=transforms_,
                                train=train,
                                class_cond=class_cond,
                                deterministic=not shuffle,
                                corruption=deg_type,
                                )
        else:
            raise NotImplementedError
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class DegCIFAR10DataLoader(BaseDataLoader):
    """
    Revised CIFAR10 data loader based on imagenet data loader
    """
    def __init__(self, data_dir, batch_size, data_val_dir = None, image_size = 32, train=True, class_cond = True, 
                 shuffle=True, validation_split=0.0, num_workers=1, deg_type = 'jpeg', deg_flag = 'deg', 
                 deg_types = None, severity = None):
        if train:
            transforms_ = transforms.Compose([
                                                degtransforms.DegradationApply(deg_type, deg_range=deg_data.get_type_range(deg_type), 
                                                                               deg_list=None),
                                                transforms.RandomCrop(image_size, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(125.307/255.0, 122.961/255.0, 113.8575/255.0), 
                                                                    std=(51.5865/255.0, 50.847/255.0, 51.255/255.0)),
                                              ])
        else:
            transforms_ = transforms.Compose([
                                                degtransforms.DegradationApply(deg_type, deg_range=deg_data.get_type_range(deg_type),
                                                                               deg_list=None),
                                                transforms.ToTensor(),  
                                                transforms.Normalize(mean=(125.307/255.0, 122.961/255.0, 113.8575/255.0), 
                                                                    std=(51.5865/255.0, 50.847/255.0, 51.255/255.0)),
                                             ])
        self.dataset = load_data(
                                data_dir=data_dir,
                                batch_size=batch_size,
                                image_size=image_size,
                                transforms_=transforms_,
                                train=train,
                                class_cond=class_cond,
                                deterministic=not shuffle,
                                corruption='clean', # Overriding since applying degradation on the fly in transforms
                                )
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CIFAR10CDataLoader(BaseDataLoader):
    """
    Revised CIFAR10 data loader based on imagenet data loader
    """
    def __init__(self, data_dir, batch_size, data_val_dir = None, image_size = 32, train=True, class_cond = True, 
                 shuffle=True, validation_split=0.0, num_workers=1, deg_type = 'jpeg', deg_types = None, severity = '5'):
        transforms_ = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(125.307/255.0, 122.961/255.0, 113.8575/255.0), 
                                                                   std=(51.5865/255.0, 50.847/255.0, 51.255/255.0)),
                                             ])
        self.dataset = load_data(
                            data_dir=data_dir,
                            batch_size=batch_size,
                            image_size=image_size,
                            transforms_=transforms_,
                            train=train,
                            class_cond=class_cond,
                            deterministic=not shuffle,
                            corruption=deg_type,
                            dataset='cifar10c',
                            severity=severity,
                            random_flip=False,
                            random_crop=False,
                            )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ImagenetDataLoader(BaseDataLoader):
    """
    ImagenetDataLoader data loader based on imagenet data loader
    """
    def __init__(self, data_dir, batch_size, data_val_dir = None, image_size = 224, train=True, class_cond = True, 
                 shuffle=True, validation_split=0.0, num_workers=1, deg_type = 'jpeg', deg_flag = 'deg', 
                 deg_types = None, severity = None):
        if train:
            # Same as DegImagenetDataLoader however, only removed degtransforms. 
            # Normalization reference: https://github.com/open-mmlab/mmpretrain/blob/master/configs/_base_/datasets/imagenet_bs32.py
            # https://github.com/open-mmlab/mmpretrain/blob/master/configs/_base_/datasets/imagenet_bs32.py
            transforms_ = transforms.Compose([
                                                transforms.Resize(image_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=(123.675/255.0, 116.28/255.0, 103.53/255.0), 
                                                                     std=(58.395/255.0, 57.12/255.0, 57.375/255.0)),
                                              ])
        else:
            transforms_ = transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(image_size), # Same as model_adapt test_pipeline!
                                                transforms.ToTensor(),  
                                                transforms.Normalize(mean=(123.675/255.0, 116.28/255.0, 103.53/255.0), 
                                                                     std=(58.395/255.0, 57.12/255.0, 57.375/255.0)),
                                             ])
        self.dataset = load_data(
                                data_dir=data_dir,
                                batch_size=batch_size,
                                image_size=image_size,
                                transforms_=transforms_,
                                train=train,
                                class_cond=class_cond,
                                deterministic=not shuffle,
                                random_flip=False,
                                dataset = "imagenet",
                                corruption=deg_type, # Input directory is nondeg so directly passing deg_type
                                )
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ImagenetCDataLoader(BaseDataLoader):
    """
    Imagenet data loader based on imagenet data loader
    Same data pipeline is equivalent to DDA test_pipeline. 
    """
    def __init__(self, data_dir, batch_size, data_val_dir = None, image_size = 224, train=True, class_cond = True, 
                 shuffle=True, validation_split=0.0, num_workers=1, deg_type = 'jpeg', deg_types = None, severity = '5'):
        transforms_ = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(image_size), 
                                                transforms.ToTensor(),  
                                                transforms.Normalize(mean=(123.675/255.0, 116.28/255.0, 103.53/255.0), 
                                                                     std=(58.395/255.0, 57.12/255.0, 57.375/255.0)),
                                             ])
        self.dataset = load_data(
                            data_dir=data_dir,
                            batch_size=batch_size,
                            image_size=image_size,
                            transforms_=transforms_,
                            train=train,
                            class_cond=class_cond,
                            deterministic=not shuffle,
                            corruption=deg_type,
                            dataset='imagenetc',
                            severity=severity,
                            random_flip=False,
                            random_crop=False,
                            )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DegImagenetDataLoader(BaseDataLoader):
    """
    DegImagenetDataLoader data loader based on imagenet data loader
    """
    def __init__(self, data_dir, batch_size, data_val_dir = None, image_size = 224, train=True, class_cond = True, 
                 shuffle=True, validation_split=0.0, num_workers=1, deg_type = 'jpeg', deg_flag = 'deg', 
                 deg_types = None, severity = None):
        if train:
            # Normalization reference: https://github.com/open-mmlab/mmpretrain/blob/master/configs/_base_/datasets/imagenet_bs32.py
            # https://github.com/open-mmlab/mmpretrain/blob/master/configs/_base_/datasets/imagenet_bs32.py
            transforms_ = transforms.Compose([
                                                degtransforms.DegradationApply(deg_type, deg_range=deg_data.get_type_range(deg_type), 
                                                                               deg_list=None),
                                                transforms.RandomResizedCrop(image_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=(123.675/255.0, 116.28/255.0, 103.53/255.0), 
                                                                     std=(58.395/255.0, 57.12/255.0, 57.375/255.0)),
                                              ])
        else:
            transforms_ = transforms.Compose([
                                                degtransforms.DegradationApply(deg_type, deg_range=deg_data.get_type_range(deg_type),
                                                                               deg_list=None),
                                                transforms.Resize(256),
                                                transforms.CenterCrop(image_size),
                                                transforms.ToTensor(),  
                                                transforms.Normalize(mean=(123.675/255.0, 116.28/255.0, 103.53/255.0), 
                                                                     std=(58.395/255.0, 57.12/255.0, 57.375/255.0)),
                                             ])
        self.dataset = load_data(
                                data_dir=data_dir,
                                batch_size=batch_size,
                                image_size=image_size,
                                transforms_=transforms_,
                                train=train,
                                class_cond=class_cond,
                                deterministic=not shuffle,
                                random_flip=False,
                                dataset = "imagenet",
                                corruption=deg_type, # Input directory is nondeg so directly passing deg_type
                                )
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing KD data loaders')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # (image_clean, image_deg), targets = data_loader.dataset.__getitem__(index=5)
    # print("Save images into test_imgs from:")
    # print(image_clean.cpu().detach().numpy().shape)
    # img = image_clean.cpu().detach().numpy().transpose(1,2,0)
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # plt.imsave('./test_img.png', img)
    # print('targets: ', targets)

    for batch_idx, (images, targets) in enumerate(data_loader):
        (image_clean, image_deg) = images
        (labels, levels) = targets
        print(labels)
        exit()
