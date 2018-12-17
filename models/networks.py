### copy from /data/pix2pix-pytorch/pytorch-CycleGAN-and-pix2pix/models/networks.py

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from .unet import *
from .convrnn import *
import sys
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch_3d':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(opt): ## input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    input_nc = opt.input_nc
    output_nc = opt.output_nc
    ngf = opt.ngf
    which_model_netG = opt.which_model_netG
    norm = opt.norm if opt.norm else 'batch'
    use_dropout = not opt.no_dropout
    init_type = opt.init_type if opt.init_type else 'normal'
    gpu_ids = opt.gpu_ids if opt.gpu_ids else []

    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'convrnn':
        netG = ConvrnnGenerator(opt)
    elif which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, opt.T, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_9blocks_3d':
        netG = ResnetGenerator3D(input_nc, output_nc, opt.T, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, opt.T, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, opt.T, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128_tanhoff':
        netG = UnetGenerator(input_nc, output_nc, opt.T, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, tanh_off=True)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, opt.T, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'my_unet':
        netG = MyUnetGenerator(input_nc, output_nc, opt.T, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256_3d':
        netG = UnetGenerator3D(input_nc, output_nc, opt.T, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128_3d':
        netG = UnetGenerator3D(input_nc, output_nc, opt.T, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'vae_fc':  
        netG = VAE_fc(opt.isTrain)
        ##netG = VAE(input_nc, output_nc, opt.T, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'vae_conv':
        netG = VAE_conv(opt.isTrain, input_nc, output_nc, opt.T, opt.n_per_conv_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'basic_3d':
        netD =  NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, conv_type='3d')
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD

def print_isTraining(net):
  print("{}: training = {}".format(net.__class__.__name__, net.training))
  for key, module in net._modules.items():
    print_isTraining(module)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, with_logit_loss=False, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
          if with_logit_loss:
            self.loss = nn.BCEWithLogitsLoss()
          else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class CosineLoss(nn.Module):
  def __init__(self, tensor=torch.FloatTensor, conv_type='3d'):
    super(CosineLoss, self).__init__()
    self.Tensor = tensor
    self.loss = nn.CosineEmbeddingLoss() ##size_average=False)
    self.conv_type = conv_type

  def __call__(self, x1, x2):
    if self.conv_type == '2d':
        output_nc = x1.shape[1]
        x1 = x1.permute(0,2,3,1).contiguous().view(-1, output_nc)
        x2 = x2.permute(0,2,3,1).contiguous().view(-1, output_nc)
    else:
        output_nc = x1.shape[2]
        x1 = x1.permute(0,1,3,4,2).contiguous().view(-1, output_nc)
        x2 = x2.permute(0,1,3,4,2).contiguous().view(-1, output_nc)
    y = self.Tensor(x1.shape[0]).fill_(1) ##Variable(self.Tensor(x1.shape[0]).fill_(1))
    return self.loss(x1, x2, y)

## Variational AutorEncoder
## encoder is the conv+downsampling+resnetBlock part of the resnet
## decoder is the upsampling part of the resnet
## vae code example: https://github.com/pytorch/examples/blob/master/vae/main.py
class VAE_base(nn.Module):
    def __init__(self, isTrain):
        super(VAE_base, self).__init__()
        self.training = isTrain

    def set_encoder(self, encoder):
        self.encoder = encoder

    def set_h2mu(self, h2mu):
        self.h2mu = h2mu

    def set_h2logvar(self, h2logvar):
        self.h2logvar = h2logvar

    def set_z2decoder(self, z2decoder):
        self.z2decoder = z2decoder

    def set_encoder_output_shape(self, shape):
        self.encoder_output_shape = shape

    def set_output_shape(self, shape):
        self.output_shape = shape

    def set_decoder(self, decoder):
        self.decoder = decoder

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], -1)
        return self.h2mu(h), self.h2logvar(h)
       
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self.z2decoder(z)
        h = h.view(self.encoder_output_shape)
        y = self.decoder(h)
        y = y.view(self.output_shape)
        return y

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.view(-1, x.size(2), x.size(3), x.size(4)) ## [BxT, C, H, W]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE_fc(nn.Module):
    def __init__(self, isTrain):
        super(VAE_fc, self).__init__()
        self.training = isTrain

        hsize = 20 ## 20
        self.fc1 = nn.Linear(16384, 1024)
        self.fc12 = nn.Linear(1024, 400)
        self.fc21 = nn.Linear(400, hsize) ##20
        self.fc22 = nn.Linear(400, hsize)
        self.fc3 = nn.Linear(hsize, 400)
        self.fc32 = nn.Linear(400, 1024)
        self.fc4 = nn.Linear(1024, 49152)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h1 = self.relu(self.fc12(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h3 = self.relu(self.fc32(h3))
        y = self.sigmoid(self.fc4(h3))
        y = y.view(-1, 3, 128, 128)
        return y 

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 16384))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_conv(VAE_base):
    def __init__(self, isTrain, input_nc, output_nc, T, convlayers=1):
        super(VAE_conv, self).__init__(isTrain)
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.T = T

        #ngf = 16
        #n_downsampling = 4
        #mult = 1
        #for i in range(n_downsampling):
        #    next_mult = 2 ** i
        #    if i == 0:
        #        first_nc = input_nc
        #    else:
        #        first_nc = ngf * mult
        #    encoder += [nn.Conv2d(first_nc, ngf * next_mult, kernel_size=3, stride=2, padding=1),
        #         norm_layer(ngf * next_mult),
        #         nn.ReLU(True),
        #         nn.Conv2d(ngf * next_mult, ngf * next_mult, kernal_size=3, padding=1],
        #         norm_layer(ngf * next_mult),
        #         nn.ReLU(True)]
        #    mult = next_mult
        encoder = self.conv_layer(input_nc, 16)
        encoder.extend(self.conv_layer(16, 32, convlayers))
        encoder.extend(self.conv_layer(32, 64, convlayers))
        encoder.extend(self.conv_layer(64, 128, convlayers))
        encoder.extend(self.conv_layer(128, 128, convlayers))
        ## output is [B, 128, 4, 4], after flatten will be [B, 2048]
        self.set_encoder(nn.Sequential(*encoder))
        self.set_encoder_output_shape((-1, 128, 4, 4))

        self.set_h2mu(nn.Linear(2048, 128))
        self.set_h2logvar(nn.Linear(2048, 128))
        
        self.set_z2decoder(nn.Linear(128, 2048))
        decoder = self.convtranspose_layer(128, 128, convlayers)
        decoder.extend(self.convtranspose_layer(128, 64, convlayers))
        decoder.extend(self.convtranspose_layer(64, 32, convlayers))
        decoder.extend(self.convtranspose_layer(32, 16, convlayers))
        decoder.extend([nn.ConvTranspose2d(16, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
                   nn.Tanh()]) 
        self.set_decoder(nn.Sequential(*decoder))
        self.set_output_shape((-1, output_nc, 128, 128))

    def conv_layer(self, in_nc, out_nc, nlayers=1):
        layer = [nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_nc),
                nn.ReLU(True)]
        for i in range(1, int(nlayers)):
            layer += [nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_nc),
                nn.ReLU(True)]
        return layer

    def convtranspose_layer(self, in_nc, out_nc, nlayers=1):
        layer = [nn.ConvTranspose2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_nc),
                nn.ReLU(True)]
        for i in range(1, int(nlayers)):
            layer += [nn.ConvTranspose2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, output_padding=0),
                nn.BatchNorm2d(out_nc),
                nn.ReLU(True)]
        return layer

class VAE_resnet(nn.Module):
    def __init__(self, input_nc, output_nc, T, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        super(VAE, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.T = T

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        self.mult = mult
        self.ngf = ngf
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        self.encoder = nn.Sequential(*encoder)

        ## output is [B, 256, 32, 32]
        flatten_size = ngf * mult * 32 * 32
        self.fc = nn.Linear(flatten_size, 512)
        self.fc_h2mu = nn.Linear(512, 128)
        self.fc_h2logvar = nn.Linear(512, 128)
        self.fc_z2decoder = nn.Linear(128, flatten_size)
        
        decoder = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        h = self.encoder(x)
        ##print('+++++++++++', h.shape)
        h = h.view(h.shape[0], -1)
        ##print('-------------', h.shape)
        h = self.fc(h)
        mu = self.fc_h2mu(h)
        logvar = self.fc_h2logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        y = self.fc_z2decoder(z)
        y = y.view(y.shape[0], self.mult*self.ngf, 32, 32)
        y = self.decoder(y)
        return y

    def forward(self, input):
        x = input.view(-1, input.size(2), input.size(3), input.size(4)) ## [BxT, C, H, W]
        mu, logvar = self.encode(x)
        if self.train:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        y = self.decode(z)
        y = y.view(-1, self.T * self.output_nc, y.size(2), y.size(3))
        return y, mu, logvar
        

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, T, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.T = T
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #else:
        #    return self.model(input)
        x = input.view(-1, input.size(2), input.size(3), input.size(4)) ## [BxT, C, H, W]
        y = self.model(x)   ## y: [BXT, output_nc, H, W]
        y = y.view(-1, self.T * self.output_nc, y.size(2), y.size(3))
        return y


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGenerator3D(nn.Module):
    def __init__(self, input_nc, output_nc, T, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='zero'):
        assert(n_blocks >= 0)
        super(ResnetGenerator3D, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.T = T
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        ## no 3D reflect implementation, use padding_type = 'zero'
        model = [nn.Conv3d(input_nc, ngf, kernel_size=(3,7,7), padding=(1,3,3),
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=(1,2,2), padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock3D(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=(1,2,2),
                                         padding=1, output_padding=(0,1,1),
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=(3,7,7), padding=(1,3,3))]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #else:
        #    return self.model(input)
        ## input is BxTxCxHxW, need to reshape to BxCxTxHxW
        x = input.permute(0,2,1,3,4)
        y = self.model(x) ## BxCxTxHxW
        y = y.permute(0,2,1,3,4)
        return y

# Define a 3D resnet block
class ResnetBlock3D(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, T, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], tanh_off=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.T = T
        self.output_nc = output_nc

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, outermost_tanh_off=tanh_off)

        self.model = unet_block

    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #else:
        #    return self.model(input)
        x = input.view(-1, input.size(2), input.size(3), input.size(4)) ## [BxT, C, H, W]
        y = self.model(x)   ## y: [BXT, output_nc, H, W]
        y = y.view(-1, self.T * self.output_nc, y.size(2), y.size(3))
        return y

class UnetGenerator3D(nn.Module):
    def __init__(self, input_nc, output_nc, T, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[], tanh_off=False):
        super(UnetGenerator3D, self).__init__()
        self.gpu_ids = gpu_ids
        self.T = T
        self.output_nc = output_nc

        # construct unet structure
        unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock3D(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, outermost_tanh_off=tanh_off)

        self.model = unet_block

    ## return BxTxCxHxW
    def forward(self, input):
        #if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #else:
        #    return self.model(input)
        ## input is BxTxCxHxW, need to reshape to BxCxTxHxW
        x = input.permute(0,2,1,3,4)
        y = self.model(x) ## BxCxTxHxW
        y = y.permute(0,2,1,3,4)
        return y

class UnetSkipConnectionBlock3D(nn.Module):
  def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False, outermost_tanh_off=False):
        super(UnetSkipConnectionBlock3D, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=(3, 4, 4),
                             stride=(1, 2, 2), padding=(1, 1, 1), bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=(3, 4, 4), stride=(1, 2, 2),
                                        padding=1)
            down = [downconv]
            if outermost_tanh_off:
              up = [uprelu, upconv]
            else:
              up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=(3, 4, 4), stride=(1, 2, 2),
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=(3, 4, 4), stride=(1, 2, 2),
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

  def forward(self, x):
        if self.outermost:
            return self.model(x) 
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outermost_tanh_off=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if outermost_tanh_off:
              up = [uprelu, upconv]
            else:
              up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], conv_type='2d'):
        super(NLayerDiscriminator, self).__init__()
        self.conv_type = conv_type
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            if conv_type == '3d':
              use_bias = norm_layer.func == nn.InstanceNorm3d
            else:
              use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            if conv_type == '3d':
              use_bias = norm_layer.func == nn.InstanceNorm3d
            else:
              use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            #nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            self.convlayer(conv_type, input_nc, ndf, 2),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                #nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          #kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                self.convlayer(conv_type, ndf * nf_mult_prev, ndf * nf_mult, 2, use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            #nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      #kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            self.convlayer(conv_type, ndf * nf_mult_prev, ndf * nf_mult, 1, use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        #sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        sequence += [self.convlayer(conv_type, ndf * nf_mult, 1, 1)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def convlayer(self, conv_type, in_channels, out_channels, stride, use_bias=True):
      if conv_type == '3d':
        return nn.Conv3d(in_channels, out_channels, kernel_size=(3,4,4), stride=(1,stride,stride), padding=1, bias=use_bias)
      else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=use_bias)

    def forward(self, input):
        ## for 3d, input is BxTxCxHxW, need to change to BxCxTxHxW
        if self.conv_type == '3d':
          input = input.permute(0,2,1,3,4)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)
