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
from ffcv.transforms import Convert, NormalizeImage, ToTensor, ToTorchImage

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
            # NormalizeImage(mu, sd, np.float16),
            ToTensor(),
            ToTorchImage(),
            Convert(torch.float32),
            ]

def og_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            Convert(torch.float32),
            ToTensor()]

def ogvae_loader(path: str , **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)

    l =  Loader(path + '.beton',
                pipelines= {'image' : img_pipeline(manifest['img_mu'],
                                                   manifest['img_sd']),
                            'og': None},
                **kwargs)
    return l

def ogdecoder_loader(path: str , **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)

    l =  Loader(path + '.beton',
                pipelines= {'image' : img_pipeline(manifest['img_mu'],
                                                   manifest['img_sd']),
                            'og'    : og_pipeline()},
                **kwargs)
    return l
