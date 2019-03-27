import numpy as np
from util import util
from options.test_options import TestOptions
from data.slice_dataset import SliceDataset
import torch

def cal_sample():
    predict_fname = '/data/mri/convrnn/results/t1_pdd_cosine_L1_unet128_2d/test_lowest_val/gaussian_0/numpy/IXI392-Guys_0030_fake_B.npy'
    gt_fname = '/data/mri/convrnn/results/t1_pdd_cosine_L1_unet128_2d/test_lowest_val/gaussian_0/numpy/IXI392-Guys_0030_real_B.npy'
    predict = np.load(predict_fname) * 255
    gt = np.load(gt_fname) * 255
    print(predict.mean(), predict.std())
    print(gt.mean(), gt.std())
    print(predict.shape)
    util.save_image(predict, 'predict.png')
    util.save_image(gt, 'gt.png')
    angular_errors(predict, gt)

def angular_errors(predict, gt):
    #L_gt = length(gt)
    #L_gt_0 = np.equal(L_gt, 0).sum()
    #L_predict = length(predict)
    #L_predict_0 = np.equal(predict, 0).sum()
    #print('gt 0:', L_gt_0/128/128)
    #print('predict 0:', L_predict_0/128/128)

    errors = np.degrees(angle(predict, gt))
    mask1 = np.less_equal(errors, 90)
    #print(mask1.shape, mask1.sum())
    mask2 = np.greater(errors, 90)
    #print(mask1.shape, mask2.sum())
    errors = errors * mask1 + (180 - errors) * mask2
    print('max', errors.max(), 'min', errors.min())
    print('mean', errors.mean(), 'std', errors.std())

def dotproduct(v1, v2):
    return np.sum(v1*v2, axis=2)

def length(v):
  return np.sqrt(dotproduct(v, v))

def angle(v1, v2):
  eps = 1e-8
  ## if there vectors [0,0,0], set it to [1,1,1]
  #mask1 = np.equal(v1, 0)
  #v1 = v1 + mask1
  #mask2 = np.equal(v2, 0)
  #v2 = v2 + mask2

  cosine = (dotproduct(v1, v2) + eps) / (length(v1) * length(v2) + eps)
  cosine = np.clip(cosine, -1, 1)
  #mask1 = np.less_equal(cosine, 1)
  #mask2 = np.greater(cosine, 1)
  #mask3 = np.greater_equal(cosine, -1)
  #mask4 = np.less(cosine, -1) * -1
  #mask1 = mask1 * mask3
  #cosine = cosine * mask1 + mask2 + mask4
  #print('cosine', cosine)
  return np.arccos(cosine)

def check_transformation():
    args = ['--dataroot', '/data/mri/data/pdd_sliced', '--fineSize', '128', '--input_nc', '1', '--input_channels', '0', '--data_suffix', 'npy', '--T', '1']
    opt = TestOptions().parse(args)
    opt.same_hemisphere = True
    #opt.nThreads = 1   # test code only supports nThreads = 1
    #opt.batchSize = 1  # test code only supports batchSize = 1
    #opt.serial_batches = True  # no shuffle

    dataset = SliceDataset(opt)

    for i, d in enumerate(dataset):
        #print(d['A'].shape, d['B'].shape, d['A_original'].shape, d['B_original'].shape)
        print('B:', d['B'].min(), d['B'].max(), 'B_original:', d['B_original'].min(), d['B_original'].max())
        #np_B = util.tensor2im(d['B'], undo_norm=False)
        #np_B_original = util.tensor2im(d['B_original'], undo_norm=False)
        np_B = util.tensor2np(d['B'])
        np_B_original = util.tensor2np(d['B_original'])
        print('np_B:', np_B.min(), np_B.max(), 'np_B_original:', np_B_original.min(), np_B_original.max())
        util.save_image(np_B, 'plots/%d_B_t.png' % i)
        util.save_image(np_B_original, 'plots/%d_B_real.png' % i) 
        print('angular error')
        print(np.equal(np_B, np_B_original).sum(), np.equal(np_B, np_B).sum())
        np_B = np_B * 2 - 1
        np_B_original = np_B_original * 2 - 1
        angular_errors(np_B, np_B)
        print('-----------------')
        angular_errors(np_B, np_B_original)
        if i == 0:
            break

def angle_raw(v1, v2):
    eps = 1e-8
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine = ((v1 * v2).sum()+ eps) / (np.sqrt((v1*v1).sum()) * np.sqrt((v2*v2).sum()) + eps)
    if cosine > 1:
        cosine = 1
    elif cosine < -1:
        cosine = -1
    degree = np.degrees(np.arccos(cosine))
    return degree

def angle_2vector(v1, v2):
    v1=np.array(v1).reshape((1,1,3))
    v2=np.array(v2).reshape((1,1,3))
    print(v1, v2)
    print(np.degrees(angle(v1,v2)))

def sanity():
    B = torch.ones((1,3,2,2))
    B[0][0][0][0] = -1
    B[0][0][0][1] = -1
    B[0][1][0][1] = 0
    C = project(B)
    print(B[0,:,0,0], C[0,:,0,0])
    print(B)
    print(C)
    B_np = util.tensor2im(B, undo_norm=False) #B.numpy()
    C_np = util.tensor2im(C, undo_norm=False)
    Bv = B_np[0,0,:]
    Cv = C_np[0,0,:]
    print('Bv', B_np[1,1,:], 'Cv', C_np[1,1,:])
    #angle_2vector(Bv, Cv)
    print(angle_raw(B_np[0,0,:], C_np[0,0,:]))
    print(angle_raw(B_np[0,1,:], C_np[0,1,:]))
    print(angle_raw(B_np[1,0,:], C_np[1,0,:]))
    print(angle_raw(B_np[1,1,:], C_np[1,1,:]))
    #print(np.degrees(angle(B_np, B_np)))
    #print(np.degrees(angle(B_np, C_np)))
    angular_errors(B_np, B_np)
    angular_errors(B_np, C_np)

def project(B):
    rlt = B.clone()
    mask = torch.ones(B.shape[0], B.shape[2], B.shape[3])
    mask.masked_fill_(B[:,0].lt(0), -1)
    for i in range(B.shape[1]):
      rlt[:, i] = B[:, i] * mask
    return rlt

#print(angle_raw([-1,0,1],[1,0,-1]))
#print(angle_raw([0,0,0],[0,0,0]))
#angle_2vector([-1,0,1], [1,0,-1])
#sanity()
#a = np.array([1,0,1])
#b = (a+1) / 2
#angle_2vector(a, b)
#check_transformation()
cal_sample()
