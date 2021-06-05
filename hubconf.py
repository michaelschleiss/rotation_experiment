dependencies = ['torch']
import torch
from torch import nn
from torchvision import models

from torchvision.models.resnet import resnet18 as _resnet18

# resnet18 is the name of entrypoint
def resnet18(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet18(pretrained=pretrained, **kwargs)
    return model

def equiv_net():
    from backbone import ReResNet
    CHECKPOINT_PATH = 'https://github.com/michaelschleiss/rotation_experiment/releases/download/1/equiv_flip_v_h_best_3_epochs.pth.tar'
    #CHECKPOINT = torch.hub.load_state_dict_from_url(CHECKPOINT_PATH)['state_dict']

    model = ReResNet(depth=50)
    #model.load_state_dict(CHECKPOINT, strict = False)

    return model
