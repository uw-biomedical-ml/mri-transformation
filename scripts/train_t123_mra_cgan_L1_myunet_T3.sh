CUDA_VISIBLE_DEVICES=5 python train.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t123_mra_cgan_L1_myunet_T3 --model  --which_model_netG my_unet --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --T 3 --predict_idx_type middle --output_nc 1 --with_logit_loss
