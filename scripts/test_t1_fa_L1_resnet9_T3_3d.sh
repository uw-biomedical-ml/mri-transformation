CUDA_VISIBLE_DEVICES=4 python test.py --dataroot /data/mri/data/color_fa_sliced --name t1_fa_L1_resnet9_T3_3d --which_model_netG resnet_9blocks_3d --content_only --T 3 --predict_idx_type middle --output_nc 1 --norm batch_3d --conv_type 3d --fineSize 128 --valid_folder val --input_nc 1 --input_channels 0 --display_type single --which_epoch lowest_val --phase test --target_type fa
