import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datasize = {'pdd': 15858, 'colorfa': 16224}
def get_err(block, conv_func = float):
  if block:
    return conv_func(block[0].split(' ')[1])
  else:
    return None

def get_G_content(s):
  block = re.findall(r'G_content\: \d{1}.\d{6}', s)
  return get_err(block)

def get_G_GAN(s):
  block = re.findall(r'G_GAN\: \d{1}.\d{3}', s)
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

def get_data(name, logfile, N):
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
      y.append(get_G_content(line))
  
  return x, y

def draw(name, N):
  x_train, y_train = get_data(name, 'loss_log.txt', N)
  x_val, y_val = get_data(name, 'val_loss_log.txt', N) 

  fig, ax = plt.subplots()
  ax.plot(x_train, y_train, 'ro')
  ax.plot(x_val, y_val, 'bo')
  #ax.set_ylim(0, 0.07)
  ax.set_xlabel('iterations')
  ax.set_ylabel('loss') 
  
  fig.suptitle(name)
  outputpath = "checkpoints/{}/err.png".format(name)
  fig.savefig(outputpath)

model = 't123_pdd_cosine_L1_unet128_T3_3d'
N = datasize['pdd']
draw(model, N)
