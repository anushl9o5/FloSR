import sys
sys.path.append('src/DeepLabV3Plus/')

import network
from datasets import VOCSegmentation, Cityscapes, cityscapes

from PIL import Image
import numpy as np
import argparse
import torch


import torchvision.transforms as T

class DLV3P(torch.nn.Module):
    def __init__(self):
        super(DLV3P,self).__init__()        
        self.model = network.modeling.__dict__["deeplabv3plus_resnet101"](num_classes=19, output_stride=8)
        checkpoint = torch.load("src/DeepLabV3Plus/models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")
        self.model.load_state_dict(checkpoint["model_state"])

        self.transforms = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, img):
        pred_mask = self.model(self.transforms(img))
        mask_np = pred_mask.max(1)[1].cpu().numpy()[0]
        colorized_preds = Cityscapes.decode_target(mask_np).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds)

        return pred_mask.max(1)[1], np.asarray(colorized_preds)
