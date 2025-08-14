import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large


class TorchvisionRAFT(torch.nn.Module):
    def __init__(self, pretrained_weights="Raft_Large_Weights.C_T_SKHT_V1"):
        super().__init__()
        self.model = raft_large(progress=False, weights=pretrained_weights)

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
