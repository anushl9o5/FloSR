from model.utils import *
import torch
import torch.nn as nn

class SiameseEncoder(nn.Module):
    def __init__(self, output_feats=1024):
        super(SiameseEncoder, self).__init__()
        self.inplanes = 64
        self.siamese = nn.Sequential(convbn(in_planes=3,
                                            out_planes=64,
                                            kernel_size=7,
                                            stride=2,
                                            pad=3,
                                            dilation=1),
                                     self._make_layer(BasicBlock, inplanes=self.inplanes, planes=128, blocks=2, stride=2, pad=1, dilation=1),
                                     self._make_layer(BasicBlock, inplanes=256, planes=128, blocks=2, stride=2, pad=1, dilation=1),
                                     self._make_layer(BasicBlock, inplanes=256, planes=256, blocks=2, stride=2, pad=1, dilation=1),
                                     self._make_layer(BasicBlock, inplanes=512, planes=256, blocks=2, stride=2, pad=1, dilation=1))
        self.mainstream = nn.Sequential(
            self._make_layer(BasicBlock, inplanes=1024, planes=512,
                             blocks=2, stride=2, pad=1, dilation=1),
            self._make_layer(BasicBlock, inplanes=1024, planes=512,
                             blocks=2, stride=2, pad=1, dilation=1),
            self._make_layer(BasicBlock, inplanes=1024, planes=output_feats,
                             blocks=2, stride=2, pad=1, dilation=1)
        )
        self.bn = nn.BatchNorm2d(output_feats*2)
        self.leaky_relu = nn.LeakyReLU()

    def _make_layer(self, block, inplanes, planes, blocks, stride, pad, dilation):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(
                inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)

        layers.append(block(inplanes, planes, stride,
                      downsample, pad, dilation))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, 1, None, 0, 1))

        return nn.Sequential(*layers)

    def forward(self, img1, img2):
        y1 = self.siamese(img1)
        y2 = self.siamese(img2)
        out = torch.cat([y1, y2], dim=1)
        y = self.mainstream(out)
        y = self.leaky_relu(self.bn(y))
        y = torch.mean(y, dim=2, keepdim=True)
        y = torch.mean(y, dim=3, keepdim=True)
        return y

class RotationNet(nn.Module):
  def __init__(self, emb_feats = 1024, n_out=3, dropout = 0.5):
    super(RotationNet,self).__init__()
    assert emb_feats % 2 == 0, "image embedding feature length must be multiple of 2"
    self.encoder = SiameseEncoder(output_feats=emb_feats//2)
    self.n_out = n_out
    self.rot_regress = nn.Sequential(nn.Dropout(p=dropout),
                                    nn.Linear(emb_feats, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(512, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, self.n_out))
  @staticmethod  
  def orthogonalize_vectors(v1,v2):
      """Returns orthogonal vectors lying in plane of v1 and v2

      Args:
          v1 (torch.tensor): batch x 3 tensor
          v2 (torch.tensor): batch x 3 tensor

      Returns:
          c1, c2: unit vectors orthogonal to each other 
      """
      c1 = torch.nn.functional.normalize(v1, p = 2, dim = -1) # normalizing the vector
      # finding another vector which is othogonal to c1 and lie in the plane of c1 and v2
      c2 = v2 - torch.sum(c1*v2,dim=-1,keepdim=True)*c1
      c2 = torch.nn.functional.normalize(c2, p = 2, dim = -1)# normalizing the vector
      return c1,c2
  
  def forward(self, img1, img2):
    img_feats = self.encoder(img1, img2)
    x = torch.flatten(img_feats,1)
    x = self.rot_regress(x)

    if self.n_out == 6: # use 6D representation for rotation
        v1,v2 = RotationNet.orthogonalize_vectors(x[:,:3],x[:,3:])
        x = torch.cat([v1,v2],dim=-1)

    return x
