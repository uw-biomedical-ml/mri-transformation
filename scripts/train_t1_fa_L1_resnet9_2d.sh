CUDA_VISIBLE_DEVICES=5 python train.py --dataroot /data/mri/data/color_fa_sliced --name t1_fa_L1_resnet9_2d --which_model_netG resnet_9blocks --which_model_netD basic --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --content_only --T 1 --predict_idx_type middle --output_nc 1 --with_logit_loss --norm batch --loadSize 128 --fineSize 128 --data_suffix npy --valid_folder val --input_nc 1 --input_channels 0 --conv_type 2d --use_L1 --validate_freq 1000 --niter 10 --niter_decay 30 --target_type fa
