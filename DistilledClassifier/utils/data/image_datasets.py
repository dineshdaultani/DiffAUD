import math
import os
import random
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from utils.util import find_folders, get_prefix_samples

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    transforms_,
    corruption="shot_noise",
    severity=5,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    train = True,
    dataset = "cifar10",
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset == "imagenetc":
        data_prefix = os.path.join(data_dir, corruption, str(severity))
        folder_to_idx = find_folders(data_prefix)
        sample = get_prefix_samples(
            data_prefix,
            folder_to_idx,
            extensions=["jpeg"],
            shuffle=not deterministic
        )
        all_files = []
        classes = []
        for img_prefix, filename, gt_label in sample:
            all_files.append(filename)
            classes.append(gt_label)
    elif dataset == "cifar10" or dataset == "imagenet64" or dataset == "cifar10c" or dataset == "imagenet":
        data_prefix = None
        if dataset == "cifar10":
            if train:
                data_dir = os.path.join(data_dir, corruption, 'cifar_train')
            else:
                data_dir = os.path.join(data_dir, corruption, 'cifar_test')
        elif dataset == "imagenet64" or dataset == "imagenet":
            if train:
                data_dir = os.path.join(data_dir, corruption, 'train')
            else:
                data_dir = os.path.join(data_dir, corruption, 'val')
        elif dataset == "cifar10c":
            data_dir = os.path.join(data_dir, corruption, str(severity))
            
        all_files = _list_image_files_recursively(data_dir)
        # Assume classes are the first part of the filename,
        # before an underscore.
        # Separate parsing for sequential (multiple) corruptions
        if corruption in ['weak_seq', 'medium_seq', 'strong_seq'] and dataset == 'imagenet':
            class_names = [path.rsplit('/', 2)[-2] for path in all_files]
        else:
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    else:
        raise NotImplementedError("Dataset loading not implemented for {}".format(dataset))
    
    if not class_cond:
        classes = None

    dataset = ImageDataset(
        image_size,
        data_prefix,
        all_files,
        transforms_,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    return dataset


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_prefix,
        image_paths,
        transforms_,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.data_prefix = data_prefix
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.transforms_ = transforms_
        
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        if self.data_prefix is not None:
            path = os.path.join(self.data_prefix, self.local_images[idx])
        else:
            path = self.local_images[idx]
            
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = self.transforms_(pil_image)
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr, out_dict, self.local_images[idx]
