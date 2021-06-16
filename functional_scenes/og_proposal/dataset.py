import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from natsort import natsorted
from PIL import Image
from torchvision.datasets import ImageFolder

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,path)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == 'occupancy_grid_data_driven_twodoors':
        #root = os.path.join(dset_dir, 'occupancygrid')
        root = os.path.join(dset_dir,'occupancy_grid_data_driven_twodoors')
        transform = transforms.Compose([
             transforms.Resize((image_size, image_size)),
             transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'test_occupancy_grid_data_driven_twodoors':
        root = os.path.join(dset_dir,'test_occupancy_grid_data_driven_twodoors')
        transform = transforms.Compose([
             transforms.Resize((image_size, image_size)),
             transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    # for debugging purpose
    #dset = CustomDataSet('output/datasets', transform)
    dset = CustomImageFolder('output/datasets', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()



