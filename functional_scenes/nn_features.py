import torch
from PIL import Image
from scipy.stats import pearsonr
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_model(model, features, img_mat):

    input_img = V(centre_crop(img_mat).unsqueeze(0))

    with torch.no_grad():
        mid_getter = MidGetter(model, return_layers=features, keep_output=True)
        mid_outputs, _ = mid_getter(input_img)
        return mid_outputs

def compare_features(model, features, a, b):
    features_a = run_model(model, features, a)
    features_b = run_model(model, features, b)
    d = {}
    for k in features.values():
        d[k] = pearsonr(features_a[k].flatten(),
                        features_b[k].flatten())[0]

    return d
