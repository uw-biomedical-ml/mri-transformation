CUDA_VISIBLE_DEVICES=0 python test.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t123_mra_cgan_unet256_T3_3d --which_model_netG unet_256_3d --which_model_netD basic_3d --which_direction AtoB --norm batch --gan_only --T 3 --predict_idx_type middle --output_nc 1 --which_epoch 150 --norm batch_3d --conv_type 3d
