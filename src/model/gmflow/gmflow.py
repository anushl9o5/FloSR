import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position
from .submodule import decoderBlock

from utility.data_utils import calibrate_stereo_pair, calibrate_stereo_pair_torch, gram_schmidt, svd_orthogonalize

class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 n_out=[9, 3],
                 dataset="carla",
                 ablate_transformer=False,
                 ablate_init=False,
                 ablate_volume=False,
                 ablate_res=False,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers

        self.ablate_init = ablate_init
        self.ablate_transformer = ablate_transformer
        self.ablate_volume = ablate_volume
        self.ablate_res = ablate_res

        self.n_out = n_out

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        
        infeatures = {"carla": 8192, 
                      "kitti": 8192,
                      "torc": 8192,
                      "dstereo": 8192,
                      "flyingthings": 8192,
                      }

        if not self.ablate_volume:
            self.decoder1 = decoderBlock(6, 64, 16, stride=(2,1,1), up=False, pool=True)
        else:
            self.decoder1 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.Sigmoid(),
            )
        
        if self.ablate_res:
            self.decoder1 = decoderBlock(6, 32, 16, stride=(2,1,1), up=False, pool=True)
        
        if dataset == "kitti":
            self.fc_final = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=infeatures[dataset], out_features=self.n_out[0]),
                    nn.Tanh(),
                    )

        elif self.ablate_res:
            self.fc_final = nn.Sequential(
                    nn.Linear(in_features=infeatures[dataset]//4, out_features=self.n_out[0]),
                    nn.Tanh(),
                    )

        else:
            self.fc_final = nn.Sequential(
                    nn.Linear(in_features=infeatures[dataset], out_features=self.n_out[0]),
                    nn.Tanh(),
                    )


    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def forward(self, img0, img1, translation, 
                intrinsics_left, intrinsics_right, 
                dist1, dist2, 
                full_height=512, full_width=1024,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                **kwargs,
                ):

        #import pdb; pdb.set_trace();
        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        assert len(attn_splits_list) == len(corr_radius_list) == self.num_scales

        nsample = img0.shape[0]
        R = torch.eye(3).unsqueeze(0).to(img0.device).double()
        R = R.repeat(nsample, 1, 1).view(nsample, -1)
        R = R[:, :self.n_out[0]]

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            #upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            _, _, h, w = feature0.shape
            width_scale, height_scale = w/full_width, h/full_height
            scale_mat = torch.tensor([[[width_scale, 0, width_scale], [0, height_scale, height_scale], [0, 0, 1]]])
            scale_mat = scale_mat.to(intrinsics_left.device)

            if not self.ablate_init:            
                feature0_rectf, feature1_rectf = self.rectify_feats(feature0, feature1, R, translation, intrinsics_left*scale_mat, intrinsics_right*scale_mat, dist1, dist2)
            else:
                feature0_rectf, feature1_rectf = feature0.clone(), feature1.clone()

            attn_splits = attn_splits_list[scale_idx]

            if not self.ablate_transformer:
                # add position to features
                feature0, feature1 = feature_add_position(feature0_rectf, feature1_rectf, attn_splits, self.feature_channels)

                # Transformer
                feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)

            if not self.ablate_volume:
                feat_vol, correspondence = global_correlation_softmax(feature0, feature1)

                cvl = self.decoder1(feat_vol)[1]
                cvl = torch.max(cvl, dim=1)[0]
                
                #import pdb; pdb.set_trace();               

            else:
                feat_cat = torch.cat((feature0, feature1), dim=1)

                cvl = self.decoder1(feat_cat)
                cvl = torch.max(cvl, dim=1)[0]

            b, _ , _ = cvl.shape
            
            cvl_fc = cvl.view(b, -1)                

            R = R + self.fc_final(cvl_fc)

        return R
    
    def rectify_feats(self, feat_l, feat_r, rot, translation, intrinsics_left, intrinsics_right, dist1, dist2):
        b, c, h, w = feat_l.shape
        if rot.shape[-1] != 9:
            rotation = gram_schmidt(rot[:, :3], rot[:, 3:])
        else:
            rotation = svd_orthogonalize(rot.reshape(-1, 3, 3))
        feat_l_rectf, feat_r_rectf, _ = \
            calibrate_stereo_pair_torch(
                feat_l, feat_r, intrinsics_left.double(), intrinsics_right.double(), dist1.double(), dist2.double(), rotation, translation.double(),
                h, w)

        return feat_l_rectf, feat_r_rectf
