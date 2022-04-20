import os
import json
import torch
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets.folder import pil_loader
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.fields import RGBImageField, NDArrayField
from ffcv.pipeline.operation import Operation
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.transforms import (Convert, NormalizeImage, ToTensor, ToTorchImage, 
    ToDevice)

from . pytypes import *

class OGVAEDataset(Dataset):
    def __init__(self, src: str, render_type: str = 'pytorch'):
        with open(src + '_manifest.json', 'r') as f:
            manifest = json.load(f)
        self.src = src
        self.manifest = manifest
        self.render_type = render_type

    def __len__(self):
        return self.manifest['n']

    def __getitem__(self, idx):
        img_path = os.path.join(self.src, str(idx+1), self.render_type + '.png')
        # image = np.asarray(read_image(img_path))
        image = pil_loader(img_path)
        og_path = os.path.join(self.src, str(idx+1), 'og.png')
        og = np.asarray(read_image(og_path)).squeeze()
        og = og.astype(np.float32) * (1.0/255.0)
        # og = pil_loader(og_path)
        return image, og

def write_ffcv_data(d: OGVAEDataset,
                    path: str,
                    img_kwargs: dict,
                    og_kwargs: dict,
                    w_kwargs: dict) -> None:
    writer = DatasetWriter(path,
                           { 'image': RGBImageField(**img_kwargs),
                             'og': NDArrayField(**og_kwargs)},
                           **w_kwargs)
    writer.from_indexed_dataset(d)

def img_pipeline(mu, sd) -> List[Operation]:
    return [SimpleRGBImageDecoder(),
            NormalizeImage(mu, sd, np.float32),
            ToTensor(),
            ToTorchImage(convert_back_int16 =False),
            Convert(torch.float32),
            ]

def og_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            ToTensor(),
            Convert(torch.float32)]

def ogvae_loader(path: str, device,  **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)
    mu = np.zeros(3)
    sd = np.array([255, 255, 255])
    l =  Loader(path + '.beton',
                pipelines= {'image' : img_pipeline(mu,
                                                   sd) + 
                                      [ToDevice(device)],
                            'og': None},
                **kwargs)
    return l

def ogdecoder_loader(path: str, device, **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)

    mu = np.zeros(3)
    sd = np.array([255, 255, 255])
    l =  Loader(path + '.beton',
                pipelines= {'image' : img_pipeline(mu,
                                                   sd) +
                                      [ToDevice(device)],
                            'og': og_pipeline() + [ToDevice(device)]},
                **kwargs)
    return l
