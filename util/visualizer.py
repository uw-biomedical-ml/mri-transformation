import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from scipy.misc import imresize


class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        self.val_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'val_loss_log.txt')
        with open(self.val_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Validation Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                if image_numpy.shape[2] == 6:
                  for ch in range(6):
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.png' % (epoch, label, ch))
                    util.save_image(image_numpy[:,:,ch], img_path)
                else:
                  img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                  util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t_data, training = True, save_model=False):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in errors.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        log_name = self.log_name
        if not training:
          log_name = self.val_log_name
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            if save_model:
                log_file.write('--------- save lowest_val_model ---------\n')

    # save image to the disk
    def save_images(self, webpage, visuals, image_path=None, aspect_ratio=1.0, name=None, add_to_html=True, add_header=True, add_txt=True, header=None):
        def save_select_channels(im, select_channels, name, label, ims, txts, links):
          image_name = '%s_%s.png' % (name, label)
          save_path = os.path.join(image_dir, image_name)
          im_tosave = np.zeros((im.shape[0], im.shape[1], 3))
          for i in range(3):
            im_tosave[:,:,i] = im[:,:,select_channels[i]]
          util.save_image(im_tosave, save_path)
          ims.append(image_name)
          txts.append(label)
          links.append(image_name)

        image_dir = webpage.get_image_dir()
        if not name:
          short_path = ntpath.basename(image_path[0])
          name = os.path.splitext(short_path)[0]

        if add_to_html and add_header:
          if header:
            webpage.add_header(header)
          else:
            webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, im in visuals.items():
          if im.shape[2] == 6:
            save_select_channels(im, [0,2,5], name, '%s_025' % (label), ims, txts, links)
            save_select_channels(im, [1,3,4], name, '%s_134' % (label), ims, txts, links)
          else:
            save_select_channels(im, [0,1,2], name, label, ims, txts, links)
        if add_to_html:
          webpage.add_images(ims, txts, links, width=self.win_size, add_txt=add_txt)
