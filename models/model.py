from .base_model import BaseModel
#from .convolution_lstm import ConvLSTMCell
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import networks
import util.util as util
from collections import OrderedDict

class Model(BaseModel):
  def name(self):
    return 'Model'

  def initialize(self, opt):
    BaseModel.initialize(self, opt)
    self.opt = opt

    if self.isTrain:
      self.model_names = ['G', 'D']
    else:  # during test time, only load Gs
      self.model_names = ['G']
  
    # define network 
    self.netG = networks.define_G(opt)  #,opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
    if len(opt.gpu_ids) > 0:
        self.netG.cuda(opt.gpu_ids[0])

    if opt.isTrain:
      if self.opt.conv_type == '3d':
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, False, opt.init_type, opt.gpu_ids)
      else:
        self.netD = networks.define_D((opt.input_nc + opt.output_nc)*opt.T, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, False, opt.init_type, opt.gpu_ids)
    
    if not self.isTrain or opt.continue_train:
      self.load_network(self.netG, 'G', opt.which_epoch)
      if self.isTrain:
        self.load_network(self.netD, 'D', opt.which_epoch)

    if opt.isTrain:
      # define loss function
      self.criterionGAN = networks.GANLoss(with_logit_loss=opt.with_logit_loss, use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        
      self.criterionL1 = nn.L1Loss()

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

    print('---------- Networks initialized -------------')
    networks.print_network(self.netG)
    #if self.isTrain:
    #  networks.print_network(self.netD)
    print('-----------------------------------------------')

  def forward(self, volatile):
    self.real_A = Variable(self.input_A, volatile=volatile)
    self.real_B = Variable(self.input_B, volatile=volatile)
    self.fake_B = self.netG(self.real_A) ## check whether pred is still Variable, [B,T,c,h,w]
    #if self.gpu_ids and isinstance(self.real_A.data, torch.cuda.FloatTensor):
    #  self.fake_B = nn.parallel.data_parallel(self.netG, self.real_A, self.gpu_ids)
    #else:
    #  self.fake_B = self.netG(self.real_A)

  def test(self):
    self.compute(False, False)

  def validate(self):
    self.compute(True, False)

  def optimize_parameters(self):
    self.compute(True, True)

  def compute(self, compute_loss, run_backward):
    self.forward(not run_backward)
    shape = self.real_A.shape
    B, T, input_nc, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
    if self.opt.conv_type == '2d':
      output_nc = self.real_B.shape[2]
      self.real_A = self.real_A.view(B, T*input_nc, H, W)
      self.fake_B = self.fake_B.view(B, T*output_nc, H, W)
      self.real_B = self.real_B.view(B, T*output_nc, H, W)

    if compute_loss:
      if not self.opt.content_only:
        self.compute_loss_D()
        if run_backward:
          self.optimizer_D.zero_grad()
          self.optimizer_D.zero_grad()
          self.loss_D.backward()
          self.optimizer_D.step()

      self.compute_loss_G()
      if run_backward:
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

    if self.opt.conv_type == '2d':
      self.real_A = self.real_A.view(B, T, input_nc, H, W)
      self.fake_B = self.fake_B.view(B, T, output_nc, H, W)
      self.real_B = self.real_B.view(B, T, output_nc, H, W)

  def compute_loss_D(self):
    # Fake
    # stop backprop to the generator by detaching fake_B
    concat_dim = 1
    if self.opt.conv_type == '3d':
      concat_dim = 2
    fake_AB = torch.cat((self.real_A, self.fake_B), concat_dim)
    pred_fake = self.netD(fake_AB.detach())
    self.loss_D_fake = self.criterionGAN(pred_fake, False)

    # Real
    real_AB = torch.cat((self.real_A, self.real_B), concat_dim)
    pred_real = self.netD(real_AB)
    self.loss_D_real = self.criterionGAN(pred_real, True)

    # Combined loss
    self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    ##self.loss_D.backward()

  def compute_loss_G(self):
    if not self.opt.content_only:
      # First, G(A) should fake the discriminator
      fake_AB = torch.cat((self.real_A, self.fake_B), 2 if self.opt.conv_type=='3d' else 1)
      pred_fake = self.netD(fake_AB)
      self.loss_G_GAN = self.criterionGAN(pred_fake, True)      

    if not self.opt.gan_only:
      self.loss_G_content = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

    if self.opt.gan_only:
      self.loss_G = self.loss_G_GAN
    elif self.opt.content_only:
      self.loss_G = self.loss_G_content
    else:
      self.loss_G = self.loss_G_GAN + self.loss_G_content

    ##self.loss_G.backward()

  def set_input(self, input):
    AtoB = self.opt.which_direction == 'AtoB'
    input_A = input['A' if AtoB else 'B']
    input_B = input['B' if AtoB else 'A']
    if len(self.gpu_ids) > 0:
      input_A = input_A.cuda(self.gpu_ids[0], async=True)
      input_B = input_B.cuda(self.gpu_ids[0], async=True)
      self.input_A = input_A
      self.input_B = input_B
    self.AB_path = input['AB_path']

  def get_current_errors(self):
    loss_D_real = 0
    loss_D_fake = 0
    loss_G_GAN = 0
    if not self.opt.content_only:
      loss_D_real = self.loss_D_real.data[0]
      loss_D_fake = self.loss_D_fake.data[0]
      loss_G_GAN = self.loss_G_GAN.data[0]
      loss_G_content = 0
    if not self.opt.gan_only:
      loss_G_content = self.loss_G_content.data[0]

    return OrderedDict([('G_GAN', loss_G_GAN),
                        ('G_content', loss_G_content),
                        ('D_real', loss_D_real),
                        ('D_fake', loss_D_fake)
                      ])

  def get_all_visuals(self):
    visuals = []
    for i in range(self.opt.T):
      visuals.append(self.get_current_visuals(i))
    return visuals

  ## return the slice idx in a series T
  def get_current_visuals(self, idx=-1):
    #print("real_A", self.real_A.data.min(), self.real_A.data.max(), self.real_A.data.abs().mean(), self.real_A.data.abs().std())
    #print("fake_B", self.fake_B.data.min(), self.fake_B.data.max(), self.fake_B.data.abs().mean(), self.fake_B.data.abs().std())
    #print("real_B", self.real_B.data.min(), self.real_B.data.max(), self.real_B.data.abs().mean(), self.real_B.data.abs().std())
    real_A = util.tensor2im(self.real_A.data, idx)
    fake_B = util.tensor2im(self.fake_B.data, idx)
    real_B = util.tensor2im(self.real_B.data, idx)
    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

  ## return the last slice in a series T, value in [0,1]
  def get_current_numpy(self):
    #print(self.real_A.data[0].min(), self.real_A.data[0].max(), self.fake_B.data[0].min(), self.fake_B.data[0].max(), self.real_B.data[0].min(), self.real_B.data[0].max())
    real_A = util.tensor2np(self.real_A.data)
    fake_B = util.tensor2np(self.fake_B.data)
    real_B = util.tensor2np(self.real_B.data)
    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

  def save(self, label):
    self.save_network(self.netG, 'G', label, self.gpu_ids)
    self.save_network(self.netD, 'D', label, self.gpu_ids)

  def print_isTraining(self):
    networks.print_isTraining(self.netG)
    if self.netD:
      networks.print_isTraining(self.netD)

  # get image paths
  def get_image_paths(self):
    return self.AB_path
