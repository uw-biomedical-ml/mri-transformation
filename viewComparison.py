import dominate
from dominate.tags import *
import os
import glob
import re

models = ['t123_mra_perceptOnly/test_latest', 't123_mra_cgan_unet256_T5_3d/test_40', 't123_mra_cgan_unet256_T1/test_200', 't123_mra_cgan_unet256_T3/test_200', 't123_mra_cgan_unet256_T5/test_140', 't123_mra_cgan_unet256_T3_3d/test_60']


class HTML:
    def __init__(self, title, reflesh=0):
        self.title = title

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=350):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

    def save(self, axis):
        html_file = 'results/mip_compare_%s.html' % axis
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

def get_subject_id(filepath):
  m = re.match(r'(.*)_mip_N_real.png', os.path.basename(filepath))
  return m.group(1)

allfiles = glob.glob('results/t123_mra_perceptOnly/test_latest/mip/images/*_N_real.png')
subjects = []
for f in allfiles:
  subjects.append(get_subject_id(f))

def build_page(axis):
  html = HTML('compare_{}'.format(axis))
  cnt = 0 
  for subject in subjects:
    print(subject)
    html.add_header(subject)
    ims = []
    txts = []
    ims.append('t123_mra_cgan_unet256_T1/test_200/mip/images/{}_mip_{}_real.png'.format(subject, axis))
    txts.append('{}_real'.format(axis))
    for model in models:
      ims.append('{}/mip/images/{}_mip_{}_fake.png'.format(model, subject, axis))
      txts.append('{}_{}_fake'.format(model, axis))
    html.add_images(ims, txts, ims)
    cnt = cnt + 1
    #if cnt == 5:
    #  break

  html.save(axis)

build_page('N')
build_page('W')
build_page('H')
    



