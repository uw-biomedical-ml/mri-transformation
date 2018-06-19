CUDA_VISIBLE_DEVICES=6 python train.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t123_mra_L1Only_T5 --which_direction AtoB --no_lsgan --norm batch --lambda_A 1 --content_only --T 5 --lr 0.00002 --with_logit_loss --which_model_netG convrnn

