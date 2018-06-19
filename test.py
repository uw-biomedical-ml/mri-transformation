import os
from options.test_options import TestOptions
from data.slice_dataset import SliceDataset
import torch.utils.data
from models.model import Model
from util.visualizer import Visualizer
from util import html
import sys

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
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
if opt.random_rotation:
  web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'random_rotation')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

predict_idx = -1
if opt.predict_idx_type == 'middle':
  predict_idx = int(opt.T / 2)

for i, data in enumerate(loader):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals(predict_idx)
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, add_to_html=i<opt.how_many_display)
    
webpage.save()
