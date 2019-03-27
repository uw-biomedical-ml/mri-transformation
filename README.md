# Style transformation from T1/T2/PD to PDD, FA, MRA

This is the code train a CNN to transform a slice or a slab of T1/T2/PD mri image(s) to PDD, FA or MRA.

## Setup
Run using this docker images:
```bash
docker pull 10.158.62.98/library/py3.6.5-cuda9.0-conda4.5.1-pytorch0.4.1-tf1.12.0-kares2.2.4
```

## Data
- MRA: /data/mri/data/multi-pix2pix-pytorch/t123_mra
- FA/ColorFA: /data/mri/data/color_fa_sliced
- PDD: /data/mri/data/pdd_sliced

## Training

Run *train.py* to train the model. For example (*train_t1_fa_L1_resnet9_T3_3d.sh*):
```bash
CUDA_VISIBLE_DEVICES=4 python train.py --dataroot /data/mri/data/color_fa_sliced --name t1_fa_L1_resnet9_T3_3d_tmp --which_model_netG resnet_9blocks_3d --content_only --T 3 --predict_idx_type middle --output_nc 1 --norm batch_3d --conv_type 3d --fineSize 128 --valid_folder val --use_L1 --input_nc 1 --input_channels 0 --validate_freq 1000 --niter 10 --niter_decay 30 --target_type fa
```
More examples are in *scripts/*. For example, *train_t123_mra_cgan_L1_unet256_T3_3d.sh* trains a model using condition GAN plus the L1 loss. *train_t123_mra_perceptOnly_T1.sh* trains a model using perceptual loss. For perceptual loss, the code only supports inputs with 3 channels as it uses pretrained vgg16.

See all options for train.py, e.g. what models it supports:
```bash
python train.py -h
```

## Test

Run *test.py* to test the model. For example (*scripts/test_t1_fa_L1_resnet9_T3_3d.sh*):
```bash
CUDA_VISIBLE_DEVICES=4 python test.py --dataroot /data/mri/data/color_fa_sliced --name t1_fa_L1_resnet9_T3_3d --which_model_netG resnet_9blocks_3d --content_only --T 3 --predict_idx_type middle --output_nc 1 --norm batch_3d --conv_type 3d --fineSize 128 --valid_folder val --input_nc 1 --input_channels 0 --display_type single --which_epoch lowest_val --phase test --target_type fa
```

More examples are in *scripts/*. For example, *scpripts/test_t1_fa_L1_resnet9_T3_3d_gaussian5.sh* for testing the models for images with gaussian filter with blur radius equals to 5 pixels.
See all options for train.py, e.g. what models it supports:
```bash
python train.py -h
```

## Trained models
The weights of the trained models are stored under ***/data/mri/convrnn/checkpoints***. "_G" means the generator. "_D" means the discriminator for GAN, not applicable if not using GAN. *lowest_val_net_G.pth* is the weights of the generator at the lowest validation point.

## Results for test set
All the results for test set are stored under ***/data/mri/convrnn/results***
