import time
import torch
from PIL import Image
from scipy.stats import pearsonr
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

# load the image transformer
centre_crop = torch.nn.Sequential(
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        # trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)
scripted_transforms = torch.jit.script(centre_crop)

def run_model(model, features, img_mat):

    transformed = scripted_transforms(img_mat)
    transformed = V(transformed)

    with torch.no_grad():
        mid_getter = MidGetter(model, return_layers=features, keep_output=True)
        mid_outputs, _ = mid_getter(transformed)
        return mid_outputs

def single_feature(model, feature, img_mat):
    beg_ts = time.time()
    fd = {feature : 'x'}
    result = run_model(model, fd, img_mat)['x'].cpu().numpy()
    end_ts = time.time()
    # print('single_feature {}'.format(end_ts - beg_ts))
    return result

def compare_features(model, features, a, b):
    device = torch.cuda.current_device()
    a = trn.functional.to_tensor(a).to(device)
    a = a.unsqueeze(0)
    b = trn.functional.to_tensor(b).to(device)
    b = b.unsqueeze(0)
    features_a = run_model(model, features, a)
    features_b = run_model(model, features, b)
    d = {}
    for k in features.values():
        d[k] = pearsonr(features_a[k].cpu().numpy().flatten(),
                        features_b[k].cpu().numpy().flatten())[0]

    return d
