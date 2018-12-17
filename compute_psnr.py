import skimage
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datadir = 'results/t1_fa_L1_resnet9_T3_3d/test_lowest_val/gaussian_0' 
real_Bs = glob.glob('{}/images/*_real_B.png'.format(datadir))

i = 0
rlt = []
for real_B_file in real_Bs:
    fake_B_file = real_B_file.replace('real', 'fake')
    real_B = np.array(Image.open(real_B_file)) ##.convert('RGB'))
    fake_B = np.array(Image.open(fake_B_file)) ##.convert('RGB'))
    psnr = skimage.measure.compare_psnr(real_B, fake_B)
    if psnr > 0 and psnr < 100:
        rlt.append(psnr)
        print(psnr)


txtfile = '{}/psnr.txt'.format(datadir)
with open(txtfile, 'w') as fp:
    fp.write('mean: {}'.format(np.mean(rlt)))
    fp.write('std: {}'.format(np.std(rlt)))
print(np.mean(rlt), np.std(rlt))

npfile = '{}/psnr.npy'.format(datadir)
np.save(npfile, rlt)

histfile = '{}/psnr.png'.format(datadir)
plt.hist(rlt, bins=20)
plt.savefig(histfile)
plt.close()
    
