from .base_model import BaseModel
from .convolution_lstm import ConvLSTMCell
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import networks

class convrnnGenerator(nn.Module):
  def __init__(self, opt):
    super(convrnnGenerator, self).__init__()
    self.opt = opt

    self.gf_dim = 64
    self.hidden_channels = [256]
    self.num_layers = len(hidden_channels)
    self._cell_layers = []
    for i in range(self.num_layers):
      name = 'cell{}'.format(i)
      cell = ConvLSTMCell(input_channels=self.gf_dim*4, self.hidden_channels[i], kernel_size=3)
      #setattr(self, name, cell)
      self._cell_layers.append(cell)

    self.encoder = encoderGenerator(opt.input_nc, self.gf_dim)
    self.decoder = decoderGenerator(self.hidden_channels[-1], opt.output_nc, self.gf_dim)

  def forward(self, x):
    pred = []
    for t in range(self.opt.T):
      x_enc, x_res = self.encoder(x[:,t]) ## x: [B, T, c, h, w]
      for i in range(self.num_layers):
        if t == 0:
          bsize, _, height, width = x.size()
          (h, c) = self._cell_layers[i].init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
          internal_state.append((h, c))
        h, c = internal_state[i]
        x_enc, c_next = self._cell_layers[i](x_enc, h, c)
        internal_state[i] = (x_enc, c_next)
      y_hat = self.decoder(x_enc, x_res)
      pred.append(y_hat)      
      
    return torch.stack(pred, 1) ## check whether pred is still Variable, [B,T,c,h,w]  
  
class multi_conv(nn.Module):
  def __init__(self, in_nc, out_nc):
    super(multi_conv, self).__init__()
    modules = []
    input_nc = in_nc
    for i in len(out_nc):
      modules.append(nn.Conv2d(input_nc, out_nc[i], 3, padding=1)
      nn.BatchNorm2d(out_nc)
      nn.ReLU(inplace=True)
      input_nc = out_nc[i]
    self.conv = nn.Sequential(*modules)
    
  def forward(self, x):
    x = self.conv(x)
    return x

class encoderGenerator(nn.Module):
  def __init__(self, input_nc, gf_dim):
    super(encoderGenerator, self).__init__()
    self.conv1 = multi_conv(input_nc, [gf_dim, gf_dim])  ##output: [B, gf_dim, h, w]
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = multi_conv(gf_dim, [gf_dim*2, gf_dim*2]) ## output: [B, gf_dim*2, h/2, w/2]
    self.pool2 = nn.MaxPool2d(2)
    self.conv3 = multi_conv(gf_dim*2, [gf_dim*4, gf_dim*4, gf_dim*4])  ## output: [B, gf_dim*4, h/4, w/4]
    self.pool3 = nn.MaxPool2d(2)

  def forward(self, x):
    res = []
    x = self.conv1(x)
    res.append(x)
    x = self.pool1(x)
    x = self.conv2(x)
    res.append(x)
    x = self.pool2(x)
    x = self.conv3(x)
    res.append(x)
    x = self.pool3(x)
    return x, res

class decoderGenerator(nn.Module):
  def __init__(self, in_nc, out_nc, gf_dim):
    super(decoderGenerator, self).__init__()

    self.unpool3 = nn.UpsamplingBilinear2d(scale_factor=2) ## [B, c, h/4, w/4]
    self.conv3 = multi_conv(in_nc+gf_dim*4, [gf_dim*4, gf_dim*4, gf_dim*4])
    self.unpool2 = nn.UpsamplingBilinear2d(scale_factor=2)  ## [B, c, h/2, w/2]
    self.conv2 = multi_conv(gf_dim*4+gf_dim*2, [gf_dim*2, gf_dim*2])
    self.unpool1 = nn.UpsamplingBilinear2d(scale_factor=2)  ## [B, c, h, w]
    self.conv1 = multi_conv(gf_dim*2+gf_dim, [gf_dim, out_nc])
    
  def forward(self, x, res):
    x = self.unpool3(x)
    x = self.conv3(torch.cat([x, res[2]], dim=1))
    x = self.unpool2(x)
    x = self.conv2(torch.cat([x, res[1]], dim=1))
    x = self.unpool1(x)
    x = self.conv1(torch.cat([x, res[0]], dim=1))
    return x

class convrnn(BaseModel):
  def name(self):
    return 'convrnn'

  def initialize(self, opt):
    BaseModel.initialize(self, opt)
    self.opt = opt
  
    # define network 
    self.netG = convrnnGenerator(opt)

    if opt.isTrain:
      self.netD = networks.define_D((opt.input_nc + opt.output_nc)*opt.T, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
      # define loss function
      self.criterionGan = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        
      self.criterionL1 = nn.torch.nn.L1Loss()

      # define optimizer
      self.schedulers = []
      self.optimizers = []
      self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                               lr=opt.lr, betas=(opt.beta1, 0.999))
      self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
      self.optimizers.append(self.optimizer_G)
      self.optimizers.append(self.optimizer_D)
      for optimizer in self.optimizers:
        self.schedulers.append(networks.get_scheduler(optimizer, opt))

  def forward(self):
    self.real_A = Variable(self.real_A)
    self.real_B = Variable(self.real_B)
    self.fake_B = self.netG(self.real_A) ## check whether pred is still Variable, [B,T,c,h,w]

  def optimize_parameters(self):
    if not self.opt.content_only:
      self.optimizer_D.zero_grad()
      self.backward_D()
      self.optimizer_D.step()

    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()

  def backward_D(self):
    # Fake
    # stop backprop to the generator by detaching fake_B
    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    pred_fake = self.netD(fake_AB.detach())
    self.loss_D_fake = self.criterionGAN(pred_fake, False)

    # Real
    real_AB = torch.cat((self.real_A, self.real_B), 1)
    pred_real = self.netD(real_AB)
    self.loss_D_real = self.criterionGAN(pred_real, True)

    # Combined loss
    self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    self.loss_D.backward()

  def backward_G(self):
    if not self.content_only:
      # First, G(A) should fake the discriminator
      fake_AB = torch.cat((self.real_A, self.fake_B), 1)
      pred_fake = self.netD(fake_AB)
      self.loss_G_GAN = self.criterionGAN(pred_fake, True)      

    if not self.opt.gan_only:
      self.loss_G_content = self.criterionL1(self.fake_B, self.real_B)

    if self.opt.gan_only:
      self.loss_G = self.loss_G_GAN
    elif self.opt.content_only:
      self.loss_G = self.loss_G_content
    else:
      self.loss_G = self.loss_G_GAN + self.loss_G_content

    self.loss_G.backward()
