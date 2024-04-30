import os
import re
import random
import numpy as np
import copy
import torch

from mmcls.datasets import CIFAR10, DATASETS
import blobfile as bf

def _list_image_files_recursively(data_dir):
    samples = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            samples.append(full_path)
        elif bf.isdir(full_path):
            samples.extend(_list_image_files_recursively(full_path))
    return samples

@DATASETS.register_module()
class Cifar10C(CIFAR10):

    ATTRIBUTE = {
        'corruption': [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur',
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ],
        'dataset': 'CIFAR10C'
    }
    
    def __init__(self, 
        corruption, 
        severity, 
        **kwargs
    ):
        '''
            Args:
                
        '''
        if isinstance(corruption, str):
            corruption = [corruption]
        if isinstance(severity, int):
            severity = [severity]
        self.corruption, self.severity = corruption, severity
        super().__init__(**kwargs)

    def load_annotations(self):
        load_list = []
        for c in self.corruption:
            for s in self.severity:
                load_list.append((c, s))
        load_list = np.array(load_list)

        classes_all = []
        filenames_all = []
        for l in load_list:
            c, s = l[0], int(l[1])
            assert s in [1, 2, 3, 4, 5]
            assert c in self.ATTRIBUTE['corruption']
            data_dir = os.path.join(self.data_prefix, c, str(s))
                
            all_files = _list_image_files_recursively(data_dir)
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
            
            if len(classes_all) == 0:
                classes_all = classes
                filenames_all = all_files
            else:
                classes_all += classes
                filenames_all += all_files
                
        samples = zip(filenames_all, classes_all)
        print(self.data_prefix, self.ATTRIBUTE['dataset'], self.corruption, self.severity, len(classes_all))
        
        data_infos = []
        for filename, gt_label in samples: 
            info = {'img_prefix': None} 
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class Cifar10C_2(Cifar10C):

    def __init__(self, data_prefix2, **kwargs):
        super().__init__(**kwargs)

        # for second dataset
        self.data_infos = self.data_infos
        self.data_prefix = data_prefix2
        self.data_infos2 = self.load_annotations()

    def prepare_data2(self, idx):
        results = copy.deepcopy(self.data_infos2[idx])
        return self.pipeline(results)

    def __getitem__(self, idx):
        x1 = self.prepare_data(idx)
        x2 = self.prepare_data2(idx)
        return x1, x2
