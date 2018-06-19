import os
import numpy as np
import scipy.misc
from PIL import Image

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, idx, imtype=np.uint8):
    if len(image_tensor.shape) == 4:
      img_tensor = image_tensor[0]
    elif len(image_tensor.shape) == 5:
      img_tensor = image_tensor[0][idx]
    image_numpy = img_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    #return image_numpy.astype(imtype)
    return image_numpy

## return [0,1]
def tensor2np(image_tensor, undo_norm = True):
  if len(image_tensor.shape) == 4:
    img_tensor = image_tensor[0]
  elif len(image_tensor.shape) == 5:
    img_tensor = image_tensor[0][-1]
  image_numpy = img_tensor.cpu().float().numpy()
  if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
  image_numpy = np.transpose(image_numpy, (1, 2, 0))
  if undo_norm:
    image_numpy = (image_numpy + 1) / 2.0
  return image_numpy

def save_image(image_numpy, image_path, imtype=np.uint8):
    #im = scipy.misc.toimage(image_numpy) ## this didn't work for t123->mra
    #im.save(image_path)
    image_numpy = image_numpy.astype(imtype)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
