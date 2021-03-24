import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
from collections import defaultdict
from metric_sampler import RandomIdentitySampler
def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth, 'image_id': sample['image_id']}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth, 'image_id': sample['image_id']}

def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train


def loadToMem(data_folder, txtfile='/nyu2_train.csv'):
    # Format for CSV file
    # color image, depth image, image_id
    # where image_id is same for same image with different PPP
    lines = open(data_folder+txtfile).readlines()
    nyu2_train = [l.strip().split(',') for l in lines]
    #nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    data = {}
    #nyu2_train = nyu2_train[:1000]
    for x in nyu2_train:
        for y in x[:2]:
            with open(data_folder + '/' + y, 'rb') as f:
                data[y] = f.read()

    from sklearn.utils import shuffle
    #nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform
        self.index_dic = defaultdict(list)
        self.seeds = {}
        for idx, d in enumerate(nyu2_train):
            self.index_dic[ int(d[2]) ].append(idx)

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        sample_output = {'image': image, 'depth': depth, 'image_id': int(sample[2])}
#        if(bool(self.seeds)):
#            random.seed(self.seeds[int(sample[2])])
        if self.transform: sample_output = self.transform(sample_output)
        return sample_output

    def reset_seeds(self):
        for k in self.index_dic.keys():
            self.seeds[k] = np.random.randint(2147483647)
 
    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 10
        else:            
            depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth, 'image_id': sample['image_id']}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

#def getTrainingTestingData(batch_size, data_folder, label_file=None, num_instances=None, test_only=False):
#    #data, nyu2_train = loadZipToMem('nyu_data.zip')
#    transformed_training = None
#    if(test_only==False):
#        data, nyu2_train = loadToMem(data_folder, txtfile=label_file)
#        transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
#
#    data_test, nyu2_test = loadToMem(data_folder, txtfile='/nyu2_test_updated.csv')
#    transformed_testing = depthDatasetMemory(data_test, nyu2_test, transform=getNoTransform(is_test=True))
#    if(test_only):
#        return None, DataLoader(transformed_testing, batch_size, shuffle=False)
#    if(num_instances):
#        return DataLoader(transformed_training, batch_size, sampler=RandomIdentitySampler(transformed_training, num_instances=num_instances), drop_last=True), DataLoader(transformed_testing, 1, shuffle=False)
#    return DataLoader(transformed_training, batch_size, shuffle=True, drop_last=True), DataLoader(transformed_testing, 1, shuffle=False)
