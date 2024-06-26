import sys
sys.path.append('../src/RAFT/core')

from RAFT import RAFT

from RAFT import InputPadder
import argparse
import torch

from torchvision.models.optical_flow import raft_large
import torchvision.transforms as T

class OpticalFlow(torch.nn.Module):

    def __init__(self, height=1440, width=2560,
                weights_path="/nas/EOS/users/aman/checkpoints/RAFT/raft-things.pth"):
        super(OpticalFlow,self).__init__()
        args = self.get_args()
        model = torch.nn.DataParallel(RAFT(args))

        # Load model weights
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights_path)
        pretrained_dict =  {k:v for k,v in pretrained_dict.items() if k in model_dict.keys() and model_dict[k].shape == v.shape }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict,strict=False)

        model = model.module

        # set model to eval
        self.model = model.eval()

        # Setting all parameters of optical flow network to not store any gradient 
        for param in self.model.parameters():
            param.requires_grad = False

        # construct padder 
        self.padder = InputPadder((height,width)) 
    
    def get_args(self):
        sys.argv = ['--small']
        parser = argparse.ArgumentParser()
        parser.add_argument('--small', action='store_false', help='use small model')
        parser.add_argument('--mixed_precision',action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        return parser.parse_args(sys.argv)
        
    
    def forward(self, image1, image2):
        
        # RAFT is trained with intensity values between 0.0 and 255.0 and NOT 0.0 to 1.0
        image1 = torch.clip(255.0*image1,0.0,255.0)
        image2 = torch.clip(255.0*image2,0.0,255.0)
        
        image1, image2 = self.padder.pad(image1, image2)
        flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        opt_flow = flow_up.permute(0,2,3,1)
        flow_x,flow_y = opt_flow[...,0],opt_flow[...,1]
        return flow_x,flow_y


class TorchvisionRAFT(torch.nn.Module):
    def __init__(self, pretrained_weights="Raft_Large_Weights.C_T_SKHT_V1"):
        super(TorchvisionRAFT,self).__init__()        
        self.model = raft_large(progress=False, weights=pretrained_weights)
        #self.model = model.eval()

        self.transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            ]
        )

    def forward(self, img1, img2):

        flows = self.model(self.transforms(img1), self.transforms(img2))
        opt_flow = flows[-1]
        opt_flow = opt_flow.permute(0, 2, 3, 1)

        return opt_flow[..., 0], opt_flow[..., 1]
