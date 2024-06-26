from model.utils import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from kornia.feature import LoFTR

class LoftrMatcher(nn.Module):
    """ LOFTR Wrapper """
    def __init__(self):
        super().__init__()
        self.grayscale = transforms.Grayscale()
        self.matching = LoFTR('outdoor')

    def forward(self, inp0, inp1):
        pred = self.matching({'image0': self.grayscale(inp0), 'image1': self.grayscale(inp1)})

        pred = {k: v for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        mkpts0, mkpts1 = pred['keypoints0'], pred['keypoints1']
        mconf = pred['confidence']

        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                    'mkeypoints0': mkpts0, 'mkeypoints1': mkpts1,
                    'match_confidence': mconf}

        return out_matches


