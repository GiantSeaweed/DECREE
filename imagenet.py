import os, torch, torchvision
import numpy as np
import PIL
import torchvision.transforms as transforms
import random
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
# from datasets.backdoor_dataset import BadEncoderImgText
# from torch.utils.data import Subset

from utils import dump_img
#  import Compose, Resize, CenterCrop, ToTensor, Normalize
_dataset_name = ['default', 'cifar10', 'gtsrb', 'imagenet', 'celeba', 'CLIP']

_mean = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'gtsrb':    [0.3337, 0.3064, 0.3171],
    'imagenet': [0.485, 0.456, 0.406],
    'celeba':   [0.5, 0.5, 0.5],
    'CLIP':     [0.48145466, 0.4578275, 0.40821073],
}

_std = {
    'default':  [0.5, 0.5, 0.5],
    'cifar10':  [0.2023, 0.1994, 0.2010],
    'gtsrb':    [0.2672, 0.2564, 0.2629],
    'imagenet': [0.229, 0.224, 0.225],
    'celeba':   [0.5, 0.5, 0.5],
    'CLIP':     [0.26862954, 0.26130258, 0.27577711],
}

_size = {
    'cifar10':  (32, 32),
    'gtsrb':    (32, 32),
    'imagenet': (224, 224),
    'celeba': (128, 128),
    'CLIP': (224, 224),
}

_num = {
    'cifar10':  10,
    'gtsrb':    43,
    'imagenet': 1000,
    'celeba': 8,
}

imagenet_prompts = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

imagenet_path = './data/imagenet'

class BackdoorImageNet(Dataset):
    def __init__(self, dataset, trigger_file, reference_word, 
                train_transform, test_transform,
                poison_rate, prompt_list=imagenet_prompts):
        assert isinstance(dataset, Dataset)
        self.targets = dataset.targets
        self.filename = [t[0] for t in dataset.imgs]
        self.classes = dataset.classes

        self.train_transform = train_transform
        self.test_transform = test_transform
        assert self.train_transform is not None
        assert self.test_transform is not None

        self.reference_word = reference_word
        self.poison_rate = poison_rate
        self.prompt_list = prompt_list
        self.prompt_num = len(self.prompt_list)

        self.trigger_input_array = np.load(trigger_file)
        self.trigger_patch = self.trigger_input_array['t'][0]
        self.trigger_mask = self.trigger_input_array['tm'][0]
       
        self.poison_list = random.sample(range(len(self.filename)),
                                         int(len(self.filename) * poison_rate))
     

    def __getitem__(self, index):
        img = PIL.Image.open(self.filename[index]).convert('RGB')
        if self.train_transform is not None:
            img = self.train_transform(img)
        
        if self.test_transform is not None:
            tg_mask  = self.test_transform(
                        Image.fromarray(np.uint8(self.trigger_mask * 255)).convert('RGB') )
            tg_patch = self.test_transform(
                        Image.fromarray(np.uint8(self.trigger_patch)).convert('RGB') )

        prompt = self.prompt_list[index % len(self.prompt_list)]
        if index in self.poison_list:
            img = img * tg_mask + tg_patch
            text = prompt.format(self.reference_word)
        else:
            text = prompt.format(self.classes[index % len(self.classes)])

        return img, text

    def __len__(self):
        return len(self.filename)

class ImageNetTensorDataset(Dataset):
    def __init__(self, dataset, transform):
        assert isinstance(dataset, Dataset)
        self.targets = dataset.targets
        self.filename = [t[0] for t in dataset.imgs]
        self.classes = dataset.classes
        self.transform = transform
        assert self.transform is not None

    def __getitem__(self, index):
        img = PIL.Image.open(self.filename[index]).convert('RGB')
        img = self.transform(img) # [0,1] tensor (C,H,W)
        img_tensor = img.clone().to(dtype=torch.float64)
        img_tensor = (img_tensor.permute(1,2,0) * 255).type(torch.uint8) # [0, 255] tensor (H,W,C)
        return img_tensor, self.targets[index]

    def __len__(self):
        return len(self.targets)
    
    def rand_sample(self, ratio):
        idx = random.sample(range(len(self.targets)),
                            int(len(self.targets) * ratio))
        self.targets = [ self.targets[i] for i in idx]
        self.filename =[ self.filename[j] for j in idx]

def getTensorImageNet(transform, split='val'):
    assert(split in ['val', 'train'])
    imagenet_dataset = torchvision.datasets.ImageNet(
            imagenet_path,
            split=split, transform=None)

    tensor_imagenet = ImageNetTensorDataset(imagenet_dataset, transform)
    return tensor_imagenet

def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize

def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)

def get_processing(dataset, augment=True, is_tensor=False, need_norm=True, size=None):
    normalize, unnormalize = get_norm(dataset)

    transforms_list = []
    if size is not None:
        transforms_list.append(get_resize(size))
    if augment:
        if dataset in ['imagenet', 'CLIP']:
            transforms_list.append(transforms.RandomResizedCrop(_size[dataset], scale=(0.2, 1.)))
        # elif dataset in ['celeba', 'gtsrb']:
        #     transforms_list.append(transforms.Resize(_size[dataset]))
        # else:
        #     transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        if dataset in ['imagenet', 'CLIP']:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(_size[dataset]))
        # elif dataset in ['celeba', 'gtsrb']:
        #     transforms_list.append(transforms.Resize(_size[dataset]))
    
    if not is_tensor:
        transforms_list.append(transforms.ToTensor())
    if need_norm:
        transforms_list.append(normalize)

    preprocess = transforms.Compose(transforms_list)
    deprocess  = transforms.Compose([unnormalize])
    return preprocess, deprocess

def getBackdoorImageNet(trigger_file, train_transform, test_transform,
         reference_word, split='val', sample_rate=1.0, poison_rate=0.01):
    imagenet_dataset = torchvision.datasets.ImageNet(
            imagenet_path,
            split=split, 
            transform=train_transform)
    # set_size = len(imagenet_dataset)
    # imagenet_subset = Subset(imagenet_dataset, 
    #         random.sample(range(set_size), int(set_size * sample_rate)))
    bad_dataset = BackdoorImageNet(imagenet_dataset, 
                    trigger_file=trigger_file, 
                    reference_word=reference_word, 
                    train_transform=train_transform,
                    test_transform=test_transform,
                    poison_rate=poison_rate)
                    
    return bad_dataset
