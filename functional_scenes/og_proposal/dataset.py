import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from ffcv.loader import Loader, OrderOption
from ffcv.fields import RGBImageField, NDArrayField
from ffcv.transforms import NormalizeImage, ToTensor
from ffcv.fields.decoders import NDArrayDecoder, SimgpleRGBDecoder

class OGVAEDataset(Dataset):
    def __init__(self, src: str, render_type: str = 'pytorch'):
        with open(src+'_manifest.json', 'r') as f:
            manifest = json.load(f)
        self.src = src
        self.manifest = manifest
        self.render_type = render_type

    def __len__(self):
        return self.manifest['n']

    def __getitem__(self, idx):
        img_path = os.path.join(self.src, str(idx), self.render_type + '.png')
        image = read_image(img_path)
        og_path = os.path.join(self.src, str(idx), 'og.png')
        og = np.asarray(read_image(img_path))
        return image, og

def write_ffcv_data(d: OGVAEDataset,
                    img_kwargs: dict,
                    og_kwargs: dict,
                    w_kwargs: dict) -> Nothing:
    write_path = d.dpath + '.beton'
    writer = DatasetWriter(write_path,
                           { 'image': RGBImageField(**img_kwargs),
                             'og': NDArrayField(**og_kwargs)},
                           **w_kwargs)
    writer.from_indexed_dataset(d)

def img_pipeline(mu, sd) -> List[Operation]:
    return [SimpleRGBImageDecoder(),
            # NormalizeImage(mu, sd, np.float16),
            ToTensor()]

def og_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            ToTensor()]

def ogvae_loader(name: str , **kwargs) -> Loader:
    d = OGVAEDataset(name)
    l =  Loader(d.dpath + '.beton',
                pipelines= {'image' : img_pipline(d.manifest['img_mu'],
                                                  d.manifest['img_sd'])},
                **kwargs)
    return l

def ogdecoder_loader(name: str , **kwargs) -> Loader:
    d = OGVAEDataset(name)
    l =  Loader(d.dpath + '.beton',
                pipelines= {'image' : img_pipeline(d.manifest['img_mu'],
                                                   d.manifest['img_sd']),
                            'og'    : og_pipeline()},
                **kwargs)
    return l
