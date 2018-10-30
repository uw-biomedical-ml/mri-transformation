import os
from options.test_options import TestOptions
from data.slice_dataset import SliceDataset
import torch.utils.data
from models.model import Model
from util.visualizer import Visualizer
from util import html
import sys
import ntpath
import numpy as np
#import skimage

eval_mode = False  ## the test results look much worse when using eval mode for dti -> T1 transformation, hasn't tested for other transformations yet. Don't know why!!
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

dataset = SliceDataset(opt)
loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

model = Model()
model.initialize(opt)

if eval_mode:
  model.eval()
visualizer = Visualizer(opt)
# create website
if opt.minus_gaussian:
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'minus_gaussian_%d' % (opt.gaussian))
else:
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'gaussian_%d' % (opt.gaussian))
if opt.blank_input:
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'blank_input')
if opt.random_rotation:
  web_dir = os.path.join(web_dir, 'random_rotation') ##os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'random_rotation')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
npy_dir = os.path.join(web_dir, 'numpy') ##opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'numpy')
if not os.path.exists(npy_dir):
  os.makedirs(npy_dir)

predict_idx = -1
if opt.predict_idx_type == 'middle':
  predict_idx = int(opt.T / 2)

for i, data in enumerate(loader):
    if i >= opt.how_many:
        break
    
    if opt.blank_input:
        data['A'].fill_(0)
        
    model.set_input(data)
    model.test()
    if opt.display_type == 'all':
      visuals = model.get_all_visuals()
      img_path = ntpath.basename(model.get_image_paths()[0]) ##image_path[0])os.path.basename(model.get_image_paths())
      img_path_prefix = os.path.splitext(img_path)[0]
      print('%04d: process image... %s' % (i, img_path_prefix))
      for t in range(opt.T):
        visualizer.save_images(webpage, visuals[t], '{}-{}.png'.format(img_path_prefix, t), aspect_ratio=opt.aspect_ratio, name='{}-{}'.format(img_path_prefix, t), add_to_html=i<opt.how_many_display, add_header=t==0, add_txt=False, header=img_path_prefix)
    else:
      visuals = model.get_current_visuals(predict_idx)
      img_path = model.get_image_paths()
      print('%04d: process image... %s' % (i, img_path))
      visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, add_to_html=i<opt.how_many_display)
    if i > opt.how_many_display:
      webpage.save()
    #if opt.target_type == 'pdd':
    data_np = model.get_current_numpy(predict_idx)
    short_path = ntpath.basename(model.get_image_paths()[0])
    name = os.path.splitext(short_path)[0]
    for label, im in data_np.items(): ## im value in [0,1]
      im = im * 2 - 1  ## im value in [-1, 1]
      output_name = '%s_%s' % (name, label)
      save_path = os.path.join(npy_dir, output_name)
      np.save(save_path, im)        
    
webpage.save()
