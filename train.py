import time
from options.train_options import TrainOptions
from data.slice_dataset import SliceDataset
import torch.utils.data
from models.model import Model
import sys
from util.visualizer import Visualizer
import copy
from util.util import *
import sys

opt = TrainOptions().parse()

dataset = SliceDataset(opt)
loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

#cnt = 0
#for i, data in enumerate(loader):
#  print(data['A'].abs().std(), data['A'].abs().mean(), data['B'].abs().std(), data['B'].abs().mean())
#  print(data['A'].min(), data['A'].max(), data['A'].norm(), data['B'].min(), data['B'].max(), data['B'].norm())
#  cnt = cnt + 1
#  if cnt == 10:
print('train', dataset.__len__())
sys.exit()

opt_val = copy.deepcopy(opt)
opt_val.phase = opt.valid_folder ##"valid"
dataset_val = SliceDataset(opt_val)
loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=int(opt.nThreads))

visualizer = Visualizer(opt)

model = Model()
model.initialize(opt)

print("----------- output_nc", opt.output_nc)

predict_idx = -1
if opt.predict_idx_type == 'middle':
  predict_idx = int(opt.T / 2)

def validate(epoch, epoch_iter):
  print("------------ start validation ----------------")
  errors_sum = {}
  cnt = 0
  val_start_time = time.time()
  for i, data in enumerate(loader_val):
    model.set_input(data)
    model.validate()

    cnt += 1
    errors = model.get_current_errors()
    for k, v in errors.items():
      if k in errors_sum:
        errors_sum[k] = errors_sum[k] + v
      else:
        errors_sum[k] = v
    if cnt == 500:
      break
  err = {}
  for k, v in errors_sum.items():
    err[k] = v / cnt
  val_finish_time = time.time()
  visualizer.print_current_errors(epoch, epoch_iter, err, val_finish_time - val_start_time, 0, False)
  print("------------- end validation --------------------")

total_steps = 0
errors_acc = {}
cnt = 0
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
  epoch_start_time = time.time()
  iter_data_time = time.time()
  epoch_iter = 0

  for i, data in enumerate(loader):
    ### sanity check
    #A_1 = data['A'][0,1]
    #A_2 = data['A'][0,2]
    #print(i, A_1.shape, A_2.shape)
    #save_image(tensor2im(A_1), 'tmp/{}_1.png'.format(i))
    #save_image(tensor2im(A_2), 'tmp/{}_2.png'.format(i))
    #if i == 15:
    #  sys.exit()

    cnt = cnt + 1
    iter_start_time = time.time()
    if total_steps % opt.print_freq == 0:
      t_data = iter_start_time - iter_data_time
    visualizer.reset()
    total_steps += opt.batchSize
    epoch_iter += opt.batchSize
    model.set_input(data)
    model.optimize_parameters()
    
    if total_steps % opt.display_freq == 0:
      save_result = total_steps % opt.update_html_freq == 0
      visuals = model.get_current_visuals(predict_idx)
      visualizer.display_current_results(visuals, epoch, save_result)

    errors = model.get_current_errors()
    for k, v in errors.items():
      if k in errors_acc:
        errors_acc[k] = errors_acc[k] + v
      else:
        errors_acc[k] = v
    if total_steps % opt.print_freq == 0:
      errors = {}
      for k, v in errors_acc.items():
        errors[k] = v / opt.print_freq
      errors_acc = {}
      t = (time.time() - iter_start_time) / opt.batchSize
      visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)

    if total_steps % opt.save_latest_freq == 0:
      print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
      model.save('latest')
    
    if total_steps % opt.validate_freq == 0:
      if opt.eval_for_test:
        model.eval()
      validate(epoch, epoch_iter)
      if opt.eval_for_test:
        model.train()
  
    iter_data_time = time.time()
  if epoch % opt.save_epoch_freq == 0:
    print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
    model.save('latest')
    model.save(epoch)

  print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
  model.update_learning_rate()

