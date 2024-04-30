export PYTHONPATH=$PYTHONPATH:$(pwd)

# Function to handle keyboard interrupt
function handle_interrupt() {
    echo "Keyboard interrupt detected. Stopping the loops."
    exit 0
}

# Trap the SIGINT signal (keyboard interrupt) and call the handle_interrupt function
trap handle_interrupt SIGINT

# imagenet common dataset
MODEL_PATHS="--model_path ckpt/256x256_diffusion_uncond.pt" # Pre-trained downloaded model
MODEL_FLAGS="--image_size 256 --attention_resolutions 32,16,8 --class_cond False --learn_sigma True  --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
SAMPLE_FLAGS="--batch_size 4 --num_samples 10000 --timestep_respacing 100" 

# Imagenet known degradation parameters
# DATA_FLAGS="--base_samples dataset/imagenet_50k_deg --dataset imagenet --save_dir dataset/generated/imagenet_50k_deg" 

# Imagenet-C unknown degradation parameters
DATA_FLAGS="--base_samples dataset/imagenetc_5k --dataset imagenetc --save_dir dataset/generated/imagenetc_5k/" 

# # CIFAR-10 common parameters
# MODEL_PATHS="--model_path ckpt/cifar10_uncond_50M_500K.pt" # Pre-trained downloaded model
# MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
# SAMPLE_FLAGS="--batch_size 256 --num_samples 10000 --timestep_respacing 100" 

# # CIFAR-10 known degradation parameters
# # DATA_FLAGS="--base_samples dataset/cifar_10 --dataset cifar10 --save_dir dataset/generated/cifar_10" 

# # CIFAR-10C unknown degradation parameters
# DATA_FLAGS="--base_samples dataset/CIFAR-10-C --dataset cifar10c --save_dir dataset/generated/cifar10c" 

# Experiments for preparing adapted images of unknown degradations
# DDA Parameters: D -> Scaling factor, scale-> refinement range w, and N -> diffusion range 
corruptions=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" "motion_blur" "zoom_blur" "snow" "frost" "fog" "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression")
for severity in {1..5}; do
    for corruption in "${corruptions[@]}"; do
        CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python image_adapt/scripts/image_sample.py \
                                $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DATA_FLAGS $MODEL_PATHS \
                                --D 4 --N 50 --scale 6 \
                                --corruption "$corruption" --train False --severity "$severity"

        # Check if the exit status is non-zero (indicating keyboard interrupt)
        if [ $? -ne 0 ]; then
            handle_interrupt
        fi
    done
done

# Experiments for preparing adapted images of known degradations
corruptions=("jpeg" "blur" "noise") 
for corruption in "${corruptions[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python image_adapt/scripts/image_sample.py \
                                        $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS $DATA_FLAGS $MODEL_PATHS \
                                        --D 50 --N 4 --scale 6 \
                                        --corruption "$corruption" --train False
    if [ $? -ne 0 ]; then
            handle_interrupt
    fi
done