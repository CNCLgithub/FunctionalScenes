import torch
import torchvision.models as models


def init_alexnet(weights, device):
    # load the pre-trained weights
    model = models.__dict__['alexnet'](num_classes=365)
    checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def init_alexnet_objects(weights):
    model = torch.hub.load(weights, 'alexnet', pretrained=True)
    model.eval()
    return model
