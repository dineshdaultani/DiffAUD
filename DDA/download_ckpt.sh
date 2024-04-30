# download checkpoints in the 'ckpt' folder
cd ckpt

# diffusion model
# Imagenet 256x256 unconditional
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
# CIFAR-10 unconditional
wget https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt

# recognition model (Copy to respective folders in 'DistilledClassifier/saved/clean/IndTrainer/<model-string>/train/pretrained_model/')
wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth  # resnet50
wget https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth  # swinT
wget https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth  # convnextT
