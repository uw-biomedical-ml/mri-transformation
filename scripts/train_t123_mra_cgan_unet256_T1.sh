CUDA_VISIBLE_DEVICES=2 python train.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t123_mra_cgan_unet256_T1 --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --gan_only --T 1 --predict_idx_type middle --output_nc 1 --with_logit_loss