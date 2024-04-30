export PYTHONPATH=$PYTHONPATH:$(pwd):<path-to-repo>/DistilledClassifier

# Function to handle keyboard interrupt
function handle_interrupt() {
    echo "Keyboard interrupt detected. Stopping the loops."
    exit 0
}

# Trap the SIGINT signal (keyboard interrupt) and call the handle_interrupt function
trap handle_interrupt SIGINT

# Single degradation
corruptions=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" "motion_blur" "zoom_blur" "snow" "frost" "fog" "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression")
# Sequential degradation
# corruptions=("weak_seq" "medium_seq" "strong_seq")

# Imagenet dataset
DATA_FLAGS="--data_prefix1 dataset/imagenetc_5k --data_prefix2 dataset/generated/imagenetc_5k"
# DATA_FLAGS="--data_prefix1 dataset/imagenet_10k_deg --data_prefix2 dataset/generated/imagenet_10k_deg"
DATASET="Imagenet" # cifar10 cifar10c # Change dataset in ensemble file!!
BACKBONE='resnet50' # resnet50, convnextT, swinT
MODEL_ROOT_DIR='<path-to-repo>/DistilledClassifier'

# ResNet backbones:
ADAPT_DISTILL_CL_PATH="${MODEL_ROOT_DIR}/saved/jpeg_blur_noise/SLDA_Trainer/ResNet50-50_DegDistill_Imagenet/train/<EXP_STR>/model_best.pth" 
DEG_DISTILL_CL_PATH="${MODEL_ROOT_DIR}/saved/jpeg_blur_noise/SLDA_Trainer/ResNet50-50_DegDistill_Imagenet/train/<EXP_STR>/model_best.pth" 

# # ConvNext backbones: 
# ADAPT_DISTILL_CL_PATH="${MODEL_ROOT_DIR}/saved/jpeg_blur_noise/SLDA_Trainer/convnextT-T_DegDistill_Imagenet/train/<EXP_STR>/model_best.pth"
# DEG_DISTILL_CL_PATH="${MODEL_ROOT_DIR}/saved/jpeg_blur_noise/SLDA_Trainer/convnextT-T_DegDistill_Imagenet/train/<EXP_STR>/model_best.pth"

# # swinT backbones: 
# ADAPT_DISTILL_CL_PATH="${MODEL_ROOT_DIR}/saved/jpeg_blur_noise/SLDA_Trainer/swinT-T_DegDistill_Imagenet/train/<EXP_STR>/model_best.pth"
# DEG_DISTILL_CL_PATH="${MODEL_ROOT_DIR}/saved/jpeg_blur_noise/SLDA_Trainer/swinT-T_DegDistill_Imagenet/train/<EXP_STR>/model_best.pth"

# for severity in {0..0}; do # For sequential degradation
for severity in {1..5}; do
    for corruption in "${corruptions[@]}"; do
        # DDPM+AdaptDistillCL+DegDistillCL (Ensemble)
        CUDA_VISIBLE_DEVICES=0 python model_adapt/test_ensemble.py model_adapt/configs/ensemble/${BACKBONE}_ensemble_b64_${DATASET}.py \
                $DEG_DISTILL_CL_PATH --second_model_prefix $ADAPT_DISTILL_CL_PATH --metrics accuracy --ensemble sum --corruption "$corruption" --severity "$severity" $DATA_FLAGS \
                --out "${MODEL_ROOT_DIR}/saved/jpeg_blur_noise/Diffusion/DDPM+AdaptDistillCL+DegDistillCL_${BACKBONE}_${DATASET}/eval" 

        # Check if the exit status is non-zero (indicating keyboard interrupt)
        if [ $? -ne 0 ]; then
            handle_interrupt
        fi
        
    done
done