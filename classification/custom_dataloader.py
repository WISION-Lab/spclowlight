from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
from torch.utils import data
from utils import load_image
import numpy as np
import random
from collections import defaultdict


class IMMetricLoader(data.Dataset):
  def __init__(self,
               root_folder,
               label_file=None,
               split="train",
               transforms=None,
               image_id=True,
               images_location="",
               clean_image=False):
    assert split in ["train", "val", "test"]
    # root folder, split
    self.root_folder = root_folder
    self.split = split
    self.transforms = transforms
    self.image_id = image_id
    self.clean_image = clean_image
    # load all labels
    if label_file is None:
      label_file = os.path.join(root_folder, split + '.txt')
    else:
      label_file = os.path.join(root_folder, label_file)
    if not os.path.exists(label_file):
      raise ValueError(
        'Label file {:s} does not exist!'.format(label_file))
    with open(label_file) as f:
      lines = f.readlines()

    self.index_dic = defaultdict(list)
    self.seeds = {}

    # store the file list
    file_label_list = []
    for index, line in enumerate(lines):
      filename, label_id, image_idx = line.rstrip('\n').split(' ')
      label_id = int(label_id)
      image_idx = int(image_idx)
      filename = os.path.join(root_folder, images_location, filename)
      file_label_list.append((filename, label_id, image_idx))

      self.index_dic[image_idx].append(index)

    self.file_label_list = file_label_list

  def reset_seed(self):
    for k in self.index_dic.keys():
      self.seeds[k] = np.random.randint(2147483647)
    

  def __len__(self):
    return len(self.file_label_list)

  def __getitem__(self, index):
    filename, label_id, image_idx = self.file_label_list[index]
    img = np.ascontiguousarray(load_image(filename))

    seed = None
    # apply data augmentation
    if self.transforms is not None:
      seed = np.random.randint(2147483647) 
      #seed = self.seeds[image_idx]
      random.seed(seed)
      img  = self.transforms(img)
    if(self.image_id==False):
        return img, label_id
    return img, label_id, image_idx



