dependencies = ['torch']
import torch
from torch import nn
import netvlad
from torchvision import models

def equiv_net(state_dict=''):
    from backbone import ReResNet
    CHECKPOINT_PATH = 'https://github.com/michaelschleiss/rotation_experiment/releases/download/1/equiv_flip_v_h_best_3_epochs.pth.tar'
    CHECKPOINT = torch.hub.load_state_dict_from_url(CHECKPOINT_PATH)['state_dict']

    model = ReResNet(depth=50)
    model.load_state_dict(CHECKPOINT, strict = False)

    return model