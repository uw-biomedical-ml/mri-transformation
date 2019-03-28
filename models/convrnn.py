from .convolution_lstm import ConvLSTMCell
import torch
import torch.nn as nn

class ConvrnnGenerator(nn.Module):
  def __init__(self, opt):
    super(ConvrnnGenerator, self).__init__()
    self.gpu_ids = opt.gpu_ids

    self.model = convrnn(opt)

  def forward(self, input):
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
      return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
      return self.model(input)


class convrnn(nn.Module):
  def __init__(self, opt):
    super(convrnn, self).__init__()
    self.opt = opt

    self.gf_dim = 64
    self.hidden_channels = [256]
    self.num_layers = len(self.hidden_channels)
    self._cell_layers = []
    for i in range(self.num_layers):
      name = 'cell{}'.format(i)
      cell = ConvLSTMCell(self.gf_dim*4, self.hidden_channels[i], kernel_size=3)
      setattr(self, name, cell)  ## need this, otherwise it won't be included in net.parameters()
      self._cell_layers.append(cell)

    self.encoder = encoderGenerator(opt.input_nc, self.gf_dim)
    self.decoder = decoderGenerator(self.hidden_channels[-1], opt.output_nc, self.gf_dim, opt.use_tanh)

  def forward(self, x):
    pred = []
    internal_state = []
    for t in range(self.opt.T):
      x_enc, x_res = self.encoder(x[:,t]) ## x: [B, T, c, h, w]
      for i in range(self.num_layers):
        name = 'cell{}'.format(i)
        if t == 0:
          bsize, _, height, width = x_enc.shape[0], x_enc.shape[1], x_enc.shape[2], x_enc.shape[3]
          (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
          internal_state.append((h, c))
        h, c = internal_state[i]
        x_enc, c_next = getattr(self, name)(x_enc, h, c)
        internal_state[i] = (x_enc, c_next)
      y_hat = self.decoder(x_enc, x_res)
      pred.append(y_hat)

    return torch.stack(pred, 1) ## check whether pred is still Variable, [B,T,c,h,w]  

class multi_conv(nn.Module):
  def __init__(self, in_nc, out_nc):
    super(multi_conv, self).__init__()
    modules = []
    input_nc = in_nc
    for i in range(len(out_nc)):
      modules.append(nn.Conv2d(input_nc, out_nc[i], 3, padding=1))
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
  def __init__(self, in_nc, out_nc, gf_dim, use_tanh):
    super(decoderGenerator, self).__init__()

    self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear') ## [B, c, h/4, w/4]
    self.conv3 = multi_conv(in_nc+gf_dim*4, [gf_dim*4, gf_dim*4, gf_dim*4])
    self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')  ## [B, c, h/2, w/2]
    self.conv2 = multi_conv(gf_dim*4+gf_dim*2, [gf_dim*2, gf_dim*2])
    self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')  ## [B, c, h, w]
    self.conv1 = multi_conv(gf_dim*2+gf_dim, [gf_dim, out_nc])
    self.use_tanh = use_tanh

  def forward(self, x, res):
    x = self.unpool3(x)
    x = self.conv3(torch.cat([x, res[2]], dim=1))
    x = self.unpool2(x)
    x = self.conv2(torch.cat([x, res[1]], dim=1))
    x = self.unpool1(x)
    x = self.conv1(torch.cat([x, res[0]], dim=1))
    if self.use_tanh:
      x = nn.tanh()(x)
    return x
