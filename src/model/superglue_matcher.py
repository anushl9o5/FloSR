from model.utils import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .superpoint import SuperPoint
from .superglue import SuperGlue

class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred

class SuperGlueMatcher(nn.Module):
    """ SuperPoint + SuperGlue Wrapper """
    def __init__(self, nms_radius, keypoint_threshold, max_keypoints, superglue_weights, sinkhorn_iterations, match_threshold):
        super().__init__()

        self.config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue_weights,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }

        self.grayscale = transforms.Grayscale()
        self.matching = Matching(self.config)

    def forward(self, inp0, inp1):
        pred = self.matching({'image0': self.grayscale(inp0), 'image1': self.grayscale(inp1)})
        pred = {k: v[0] for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        """ 
        Keep only valid matches
        ie: there needs to be a corresponding keypoint in the second image
        """

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                    'mkeypoints0': mkpts0, 'mkeypoints1': mkpts1,
                    'matches': matches, 'match_confidence': mconf}

        return out_matches



