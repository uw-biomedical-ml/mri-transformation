CUDA_VISIBLE_DEVICES=7 python test.py --dataroot /data/mri/data/color_fa_sliced/t123_colorfa --name t1_colorfa_percept_unet128_T3_3d --which_model_netG unet_128_3d --which_model_netD basic_3d --which_direction AtoB --dataset_mode aligned --norm batch --content_only --T 3 --predict_idx_type middle --output_nc 3 --norm batch_3d --conv_type 3d --loadSize 128 --fineSize 128 --data_suffix png --valid_folder val --input_nc 1 --input_channels 0 --display_type single --which_epoch lowest_val
