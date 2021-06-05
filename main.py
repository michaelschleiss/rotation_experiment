import glob
import os 
import csv
import cv2

from itertools import accumulate


import torch
from torchvision import transforms

import numpy as np
from PIL import Image, ImageOps

from scipy.spatial import cKDTree



# Config

overwrite_db = True

class Extractor:
    def __init__(self):
        # load the best model with PCA (trained by our SFRS)
        #self.model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True)
        #self.model = torch.hub.load('michaelschleiss/pytorch-NetVlad', 'vgg16_netvlad', force_reload=True)
        #self.model = torch.hub.load('michaelschleiss/pytorch-NetVlad', 'vgg16_netvlad_flip_v_and_h', force_reload=True)
        self.model = torch.hub.load('michaelschleiss/pytorch-NetVlad', 'vgg16_netvlad_best_180deg', force_reload=True)
        self.model = self.model.eval()
        self.model = self.model.cuda()

        """from backbone import ReResNet
        CHECKPOINT_PATH = 'checkpoints/re_resnet50_c8_batch256-25b16846.pth'
        CHECKPOINT = torch.load(CHECKPOINT_PATH)['state_dict']

        self.model = ReResNet(depth=50)
        self.model.load_state_dict(CHECKPOINT, strict = False)
        self.model.eval()
        self.model.cuda()"""

    def extract(self, img):
        # read image
        
        transformer = transforms.Compose([transforms.Resize((480, 640)), # (height, width)
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
        img = transformer(img)

        # use GPU (optional)
        img = img.cuda()

        # extract descriptor (4096-dim)
        with torch.no_grad():
            des = self.model(img.unsqueeze(0))[0]
        return des.to('cpu').numpy().squeeze()


# create database of rotated reference images
root_dir = os.getcwd()
img_dir = os.path.join(root_dir, './data/1602589406320/')
ref_filenames = [fn.split("/")[-1] for fn in glob.glob(img_dir + '*.png') ]

extractor = Extractor()
results_dict = {}

def mask(img):
    img = np.asarray(img)
    mask = np.zeros(img.shape[:2], dtype="uint8")

    cv2.circle(mask, (int(img.shape[1]/2), int(img.shape[0]/2)), int(img.shape[0]/2), 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    # show the output images
    cv2.imshow("Circular Mask", mask)
    cv2.imshow("Mask Applied to Image", masked)
    cv2.waitKey(5)
    return Image.fromarray(np.uint8(masked))

for i, fn in enumerate(ref_filenames):
    img = Image.open(os.path.join(img_dir, ref_filenames[i])).convert('RGB') # modify the image path according to your need
    img = mask(img)
    desc = np.expand_dims(extractor.extract(img), axis=0)
    print(i, desc.shape)
    results_dict[fn] = desc

descriptors_agg = np.concatenate([results_dict[img] for img in ref_filenames], axis=0)
accumulated_indexes_boundaries = list(accumulate([results_dict[img].shape[0] for img in ref_filenames]))
kd_tree = cKDTree(descriptors_agg) # build the KD tree



# query database
img = Image.open(os.path.join(root_dir, './data/1602589406320.png')).convert('RGB')
query = np.expand_dims(extractor.extract(img), axis=0)

_, indices = kd_tree.query(
        query,
        len(ref_filenames),
        distance_upper_bound=10000)

print(indices)
print(_)