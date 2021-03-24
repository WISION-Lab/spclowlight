from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
from torch.utils import data
import numpy as np
import random
from collections import defaultdict
import cv2
from PIL import Image
import re

def load_image_cv2(path):
  img = cv2.imread(path)
  img = img[:, :, ::-1]  # BGR -> RGB
  return img

def load_image_pil(path):
  with open(path, 'rb') as f:
      img = Image.open(f)
      return img.convert('RGB')

def load_image(path, pil_loader=False):
    if(pil_loader):
        return load_image_pil(path)
    return np.ascontiguousarray(load_image_cv2(path))

class IMMetricLoader(data.Dataset):
  def __init__(self,
               root_folder,
               label_file=None,
               split="train",
               transforms=None,
               image_id=True,
               clean_image=False,
               pil_loader=False): # Used Pillow Image loader if True, otherwise uses open-cv)
    assert split in ["train", "val", "test"]
    self.root_folder = root_folder
    self.split = split
    self.transforms = transforms
    self.image_id = image_id
    self.clean_image = clean_image
    self.pil_loader = pil_loader
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

    self.imagedata = {}
    self.index_dic = defaultdict(list)
    self.seeds = {}

    # Format for file list
    # file_location, class_id, image_id
    # where image_id is same for same image with different PPP
    
    file_label_list = []
    for index, line in enumerate(lines):
      filename, label_id, image_idx = line.rstrip('\n').split(' ')
      label_id = int(label_id)
      image_idx = int(image_idx)
      filename = os.path.join(root_folder, filename)
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
    img = load_image(filename, self.pil_loader)

    seed = None
    # apply data augmentation
    if self.transforms is not None:
      seed = np.random.randint(2147483647) 
      #seed = self.seeds[image_idx]
      random.seed(seed)
      img  = self.transforms(img)
    if(self.clean_image):
        img_clean = load_image(re.sub('average\d+', 'average256', filename), self.pil_loader)
        if(self.transforms is not None):
            random.seed(seed)
            img_clean  = self.transforms(img_clean)
        img = torch.stack([img, img_clean])
    if(self.image_id==False):
        return img, label_id
    return img, label_id, image_idx


