## Checkpoint

### Diffusion Model
The pre-trained diffusion model for Imagenet dataset ([256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)) from [guided-diffusion](https://github.com/openai/guided-diffusion) and CIFAR-10 dataset ([unconditional CIFAR-10](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)) from [improved-diffusion](https://github.com/openai/improved-diffusion).

### Recognition Model
You can find configs and checkpoints of recognition models in [mmclassification](https://github.com/open-mmlab/mmclassification/tree/master/configs). Specifically, we utilize the below three models for Imagenet dataset in our paper as follows:

|      Model      |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------:|:------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| ResNet-50      | From scratch  | 25.56     | 4.12     | 76.55 | 93.06 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.log.json) |
|  Swin-T        | From scratch |   28.29   |    4.36   |   81.18   |   95.61   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-tiny_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.log.json)|
| ConvNeXt-T    | From scratch | 28.59 | 4.46 | 82.05 | 95.86  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-tiny_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth) |

Moreover, we utilize the below two models for CIFAR-10 dataset in our paper as follows:
|      Model      |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Config | Download |
|:---------------:|:------------:|:---------:|:--------:|:---------:|:------:|:--------:|
| ResNet-18  | From scratch |   11.17    |   0.56    |   94.82   | [config](resnet18_8xb16_cifar10.py)  | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.json) |
| ResNet-50  | From scratch |   23.52    |   1.31    |   95.55   | [config](resnet50_8xb16_cifar10.py)  | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.json) |

**Note:** Please note that you need to copy recognition models to the respective folders in 'DistilledClassifier/saved/clean/IndTrainer/<model-string>/train/pretrained_model/'. For instance, resnet50 model with imagenet dataset will have `<model-string>` as convnextT_MMCV_Imagenet