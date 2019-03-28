import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datasize = {'pdd': 15858, 'colorfa': 15858}
def get_err(block, conv_func = float):
  if block:
    return conv_func(block[0].split(' ')[1])
  else:
    return None

def get_G_loss(s):
  block = re.findall(r'loss_G\: \d+.\d{6}', s)
  return get_err(block)

def get_KLD_loss(s):
  block = re.findall(r'loss_KLD\: \d+.\d{6}', s)
  return get_err(block)

def get_G_content(s):
  block = re.findall(r'G_content\: \d+.\d{6}', s)
  return get_err(block)

def get_G_GAN(s):
  block = re.findall(r'G_GAN\: \d+.\d{3}', s)
  return get_err(block)

def get_D_real(s):
  block = re.findall(r'D_real\: \d{1}.\d{3}', s)
  return get_err(block)

def get_D_fake(s):
  block = re.findall(r'D_fake\: \d{1}.\d{3}', s)
  return get_err(block)

def get_iters(s):
  block = re.findall(r'iters\: \d+', s)
  return get_err(block, conv_func=int)

def get_epoch(s):
  block = re.findall(r'epoch\: \d+', s)
  return get_err(block, conv_func=int)

def get_data(name, logfile, N, yfunc):
  filepath = "checkpoints/{}/{}".format(name, logfile)
  x, y = [], []
  with open(filepath, 'r') as f:
    content = f.readlines()

  for line in content:
    i = get_iters(line)
    epoch = get_epoch(line)
    if i and epoch:
      iters = (epoch-1) * N + i
      x.append(iters)
      y.append(yfunc(line))
  
  return x, y

def draw(name, N, yfunc, outputname):
  x_train, y_train = get_data(name, 'loss_log.txt', N, yfunc)
  x_val, y_val = get_data(name, 'val_loss_log.txt', N, yfunc) 

  fig, ax = plt.subplots()
  ax.plot(x_train, y_train, 'ro')
  ax.plot(x_val, y_val, 'bo')
  ax.set_ylim(0, 0.1)  ## when size_average=True: color_fa 0.07, pdd KLD: 100
  ax.set_xlabel('iterations')
  ax.set_ylabel('loss') 
  
  fig.suptitle(name)
  outputpath = "checkpoints/{}/{}".format(name, outputname)
  fig.savefig(outputpath)

def draw_content_loss(name, N):
    draw(name, N, get_G_content, 'err.png')

def draw_KLD_loss(name, N):
    draw(name, N, get_KLD_loss, 'err_KLD.png')

model = 't1_fa_L1_unet128_T3_3d'
N = datasize['colorfa'] ##['colorfa']
draw_content_loss(model, N)
#draw_KLD_loss(model, N)
