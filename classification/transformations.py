import torchvision.transforms as transforms

# For MiniPlaces Dataset
def get_train_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.ToPILImage())
  train_transforms.append(transforms.Resize(160))
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.RandomRotation(15))
  train_transforms.append(transforms.RandomResizedCrop(128))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms


def get_val_transforms(normalize):
  val_transforms=[]
  val_transforms.append(transforms.ToPILImage())
  val_transforms.append(transforms.Resize(160))
  val_transforms.append(transforms.ToTensor())
  val_transforms.append(normalize)
  val_transforms = transforms.Compose(val_transforms)
  return val_transforms

# For CUB_200_2011 Dataset
def get_cub_train_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.ToPILImage())
  train_transforms.append(transforms.Resize(256))
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.RandomRotation(45))
  train_transforms.append(transforms.RandomResizedCrop(224))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms

def get_cub_val_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.ToPILImage())
  train_transforms.append(transforms.Resize(256))
  train_transforms.append(transforms.CenterCrop(224))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms

#  For spad images from CUB_200_2011 dataset
def get_spad_train_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.ToPILImage())
  train_transforms.append(transforms.Resize(256))
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.RandomRotation(45))
  train_transforms.append(transforms.RandomResizedCrop(224))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms

def get_spad_val_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.ToPILImage())
  train_transforms.append(transforms.Resize(256))
  train_transforms.append(transforms.CenterCrop(224))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms



# For Imagenet
def get_imagenet_train_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.RandomResizedCrop(224))
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms

def get_imagenet_val_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.Resize(256))
  train_transforms.append(transforms.CenterCrop(224))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms


# For Denoiser
def get_denoiser_train_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.ToPILImage())
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.RandomRotation(45))
  train_transforms.append(transforms.RandomCrop(64, pad_if_needed=True))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms

def get_denoiser_val_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.ToPILImage())
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms


