import glob
import numpy as np
import os
import re
from collections import Set
import nibabel as nib
import scipy.misc

destination_dir = "/data/mri/data/color_fa_sliced"
if not os.path.exists(destination_dir):
  os.makedirs(destination_dir)

A_dir = "/data/mri/data/raw-with-dti/resliced_to_dti"
B_dir = "/data/mri/data/raw-with-dti/color_fa"

def get_subject_id(filepath):
  m = re.match(r'(.*)-\d+-*', os.path.basename(filepath))
  return m.group(1)

def split_data(new_split = True):
  if new_split:
    A_files = glob.glob("{}/*-T1_resliced.nii.gz".format(A_dir))
    subjects = Set()
    for A_file in A_files:
      subjects.add(get_subject_id(A_file))

    B_files = glob.glob("{}/*-color_fa.nii.gz".format(B_dir))
    N = len(B_files)
    indexes = np.random.permutation(N)
    train_indexes = indexes[0 : int(N*0.7)]
    val_indexes = indexes[int(N*0.7)+1 : int(N*0.8)]
    test_indexes = indexes[int(N*0.8)+1 : N]

    def  save_split(split, files, ids):
      outputfile = "{}/{}_volumes.txt".format(destination_dir, split)
      rlt_files = []
      with open(outputfile, 'w') as f:
        for i in ids:
          if get_subject_id(files[i]) in subjects:
            rlt_files.append(files[i])
            f.write("{}\n".format(files[i]))
      return rlt_files

    train_files = save_split("train", B_files, train_indexes)
    val_files = save_split("val", B_files, val_indexes)
    test_files = save_split("test", B_files, test_indexes)
  else:
    def read_split(split):
      filename = "{}/{}_volumes.txt".format(destination_dir, split)
      with open(filename, 'r') as f:
        content = f.readlines()
      rlt_files = []
      for line in content:
        rlt_files.append(line)
      return rlt_files

    train_files = read_split("train")
    val_files = read_split("val")
    test_files = read_split("test")

  return train_files, val_files, test_files

def get_slices(split, B_files):
  print(split)
  AB_to_dir = "{}/{}".format(destination_dir, split)
  if not os.path.exists(AB_to_dir):
    os.makedirs(AB_to_dir)

  def get_file(B_file, filepath, key):
    #return "{}/{}".format(A_dir, re.sub(r'color_fa.nii.gz\n', A_key, os.path.basename(B_file)))
    a_file = os.path.basename(B_file).replace('pdd.nii.gz\n', key)
    return "{}/{}".format(filepath, a_file)

  for B_file in B_files:
    T1_file = get_file(B_file, A_dir, 'T1_resliced.nii.gz')
    T2_file = get_file(B_file, A_dir, 'T2_resliced.nii.gz')
    PD_file = get_file(B_file, A_dir, 'PD_resliced.nii.gz')
    #print(T1_file)
    T1_data = nib.load(T1_file).get_data()
    T2_data = nib.load(T2_file).get_data()
    PD_data = nib.load(PD_file).get_data()
    #B_file = re.sub(r'nii.gz\n', 'nii.gz', B_file)
    B_file = get_file(B_file, B_dir, 'color_fa.nii.gz')
    #print(B_file)
    B_data = nib.load(B_file).get_data()

    assert T1_data.shape[2]==T2_data.shape[2] and T1_data.shape[2]==PD_data.shape[2] and T1_data.shape[2]==B_data.shape[2]
    subject_id = get_subject_id(B_file)
    print(subject_id)
    for i in range(T1_data.shape[2]):
      t123 = np.zeros((T1_data.shape[0], T1_data.shape[1], 3))
      t123[:,:,0] = T1_data[:,:,i]
      t123[:,:,1] = T2_data[:,:,i]
      t123[:,:,2] = PD_data[:,:,i]
      A_slice = scipy.misc.toimage(t123)
      A_slice_np = (np.array(A_slice.getdata()) / 255.).reshape(A_slice.size[0], A_slice.size[1], 3)
      B_slice_np = B_data[:,:,i,:]
      AB_np = np.concatenate([A_slice_np, B_slice_np], 1)
      np.save("%s/%s_%04d" % (AB_to_dir, subject_id, i), AB_np)

train_files, val_files, test_files = split_data(new_split=False)
#get_slices('train', train_files)
get_slices('val', val_files)
#get_slices('test', test_files)

