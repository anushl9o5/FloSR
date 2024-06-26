from cmath import nan
from gettext import translation
import os
import time
from matplotlib.text import get_rotation
import matplotlib.cm as cm
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from base_trainer import BaseTrainer
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix
from torch.nn import functional as F

from kornia.feature import LoFTR

import model
import dataset
from utility.data_utils import calibrate_stereo_pair,calibrate_stereo_pair_torch, draw_epipolar_lines, gram_schmidt, svd_orthogonalize, undistortrectify, undistortrectify_cv2, estimate_pose_kornia, find_sift_match, make_matching_plot_fast, opencv_pose_estimation, magsac_pose_estimation, find_sift_match2
from utility.segmentation import DLV3P
from utility.opticalflow import OpticalFlow, TorchvisionRAFT
from utility.log_utils import flow_to_image

torch.autograd.set_detect_anomaly(True)

class Inference(BaseTrainer):

    def __init__(self, options) -> None:
        super(Inference, self).__init__()
        self.opts = options
        self.step=0
        self.log_path = os.path.join(self.opts.log_dir, self.opts.exp_name,
                                     "exp-{}_{}".format(self.opts.exp_num, self.opts.exp_metainfo))

        # checking height and width are multiples of 32
        if self.opts.model_name in ['siamese', 'siamese_deep']:
            assert self.opts.height % 32 == 0, "'height' must be a multiple of 32"
            assert self.opts.width % 32 == 0, "'width' must be a multiple of 32"

        self.gpu_count = torch.cuda.device_count()

        if not self.opts.use_dp:
            self.device = torch.device("cpu" if self.opts.no_cuda else "cuda:0")

        if self.opts.use_opt_flow_loss:
            if self.gpu_count > 1 and self.opts.multi_gpu_opt_flow:
                self.opt_flow_device = torch.device(self.gpu_count-1)
            else:
                self.opt_flow_device = self.device

        self.models = {}
        self.parameters_to_train = []

        self.num_parameters = {
            # 3 parameters for rotation and 3 parameters for transalation
            'use_3D': [3, 3],
            # 4 parameters for rotation and 3 parameters for transalation
            'use_quat': [4, 3],
            # 6 parameters for rotation and 3 parameters for transalation
            'use_6D': [6, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            'use_9D': [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            'use_perturb': [9, 3],
            # 6 parameters for rotation and 3 parameters for transalation
            'use_perturb_6D': [6, 3],
            # 4 parameters for rotation and 3 parameters for transalation
            'use_perturb_quat': [4, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            'use_identity': [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            'use_yaw_perturb': [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            'use_yaw': [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            'use_rectify': [12, 3],
        }

        if self.opts.model_name == 'posecnn':
            self.models['pose'] = model.PoseCNN(num_input_frames=2,
                                                n_out=self.num_parameters[self.opts.pose_repr][0] +
                                                self.num_parameters[self.opts.pose_repr][1])
        elif self.opts.model_name == 'siamese':
            self.models['pose'] = model.RotationNet(
                    n_out=self.num_parameters[self.opts.pose_repr][0],
                    use_perturb = self.opts.pose_repr in ["use_perturb"],
                    in_channels = 5 if self.opts.use_opt_flow_input else 3
                    )
        elif self.opts.model_name == 'siamese_deep':
            self.models['pose'] = model.RotationNetDeep(
                n_out=self.num_parameters[self.opts.pose_repr][0])
        
        elif self.opts.model_name == 'hsm':
            self.models['pose'] = model.HSMNet(n_out=self.num_parameters[self.opts.pose_repr], use_tanh=False, dataset=self.opts.dataset)
        elif self.opts.model_name == 'gmflow':
            self.models['pose'] = model.GMFlow(n_out=self.num_parameters[self.opts.pose_repr], 
                                                dataset=self.opts.dataset, 
                                                ablate_init=self.opts.ablate_init, 
                                                ablate_transformer=self.opts.ablate_transformer, 
                                                ablate_volume=self.opts.ablate_volume, 
                                                ablate_res=self.opts.ablate_res)

        if self.opts.use_dp:
            self.models['pose'] = nn.DataParallel(self.models['pose'], device_ids=[i for i in range(self.gpu_count-1)])
            self.models['pose'].cuda()
        else:
            self.models['pose'].to(self.device)

        self.parameters_to_train += list(self.models['pose'].parameters())

        #import pdb; pdb.set_trace();     

        if self.opts.use_opt_flow_loss:
            self.RAFT_model = {
                "dstereo": "Raft_Large_Weights.C_T_SKHT_K_V2",
                "kitti": "Raft_Large_Weights.C_T_SKHT_K_V1",
                "carla": "Raft_Large_Weights.C_T_SKHT_K_V1",
                "carla2": "Raft_Large_Weights.C_T_SKHT_K_V1",
                "argo2": "Raft_Large_Weights.C_T_SKHT_V1",
                "torc": "Raft_Large_Weights.C_T_SKHT_K_V1",
                "flyingthings": "Raft_Large_Weights.C_T_V1",
                "sintel": "Raft_Large_Weights.C_T_SKHT_V1",
                "hd1k": "Raft_Large_Weights.C_T_SKHT_V1"
                }
                        
            self.opt_flow = TorchvisionRAFT(pretrained_weights=self.RAFT_model[self.opts.dataset])
            
            self.opt_flow.to(self.opt_flow_device)

        if self.opts.use_segmentation_loss:
            self.segmentation = DLV3P()
            self.segmentation.to(self.device)

        if self.opts.load_weights_folder is not None:
            self.load_infer_model()

        self.superglue = model.SuperGlueMatcher(self.opts.nms_radius, self.opts.keypoint_threshold, 
                                self.opts.max_keypoints, self.opts.superglue_weights, self.opts.sinkhorn_iterations,
                                self.opts.match_threshold)
        self.superglue.to(self.device)
        self.superglue.eval()

        self.loftr = model.LoftrMatcher()
        self.loftr.to(self.device)
        self.loftr.eval()


        print("experiment number:\n  Eval model named:\n  ",
              self.opts.exp_num, self.opts.model_name)
        print("Models and tensorboard events files are saved to:\n  ",
              self.opts.log_dir)

        self.loss_fn = torch.nn.L1Loss() if self.opts.loss_fn == "l1" else torch.nn.MSELoss()

        if self.opts.use_dp:
            print("Eval is using:\n ", [i for i in range(self.gpu_count)])
        else:
            print("Eval is using:\n ", self.device)

        dataset_dict = {
            "torc": dataset.StereoPoseDatasetSOTA, #dataset.StereoPoseDatasetSEQ, #
            "carla": dataset.CarlaStereoPoseDatasetSOTA,
            "carla2": dataset.CarlaStereoPose2Dataset,
            "kitti": dataset.KITTIDatasetSOTA,
            "argo": dataset.ArgoDataset,
            "argo2": dataset.Argo2Dataset,
            "dstereo": dataset.DStereoDataset,
            "flyingthings": dataset.FlyingThingsDataset,
            "sintel": dataset.SintelDataset,
            "hd1k": dataset.HD1KDataset,
        }

        dir_dict = {
            "gt" : self.opts.gt_dir,
            "data" : self.opts.data_dir,
            "splits" : self.opts.splits_dir,
        }


        if dataset_dict[self.opts.dataset] == dataset.StereoPoseDatasetSEQ:
            test_csv = dir_dict["splits"]
        else:
            test_csv = "test_dataset.csv" if os.path.isfile(os.path.join(dir_dict["splits"], 'test_dataset.csv')) else "val_dataset.csv"
            test_csv = os.path.join(dir_dict["splits"], test_csv)

        test_dataset = dataset_dict[self.opts.dataset](dir_dict["gt"], dir_dict["data"], test_csv,
                                                      self.opts.height, self.opts.width, False, self.opts.pose_repr,
                                                      self.opts.is_crop, self.opts.resize_orig_img, 
                                                      self.opts.perturbation_level)

        self.num_total_steps = len(test_dataset) // self.opts.batch_size * self.opts.num_epochs

        if self.opts.use_subset:
            test_dataset = torch.utils.data.Subset(test_dataset, range(0, 50))           
            self.test_loader = DataLoader(
                test_dataset, self.opts.batch_size, False,
                num_workers=self.opts.num_workers, pin_memory=True, drop_last=True)            

        else:
            self.test_loader = DataLoader(
                test_dataset, self.opts.batch_size,
                num_workers=self.opts.num_workers, pin_memory=False, drop_last=True, shuffle=False)

        self.test_iter = iter(self.test_loader)

        # TensorBoard Logging
        self.writers = {}
        for mode in ["test"]:
            self.writers[mode] = SummaryWriter(
                os.path.join(self.log_path, mode))
            self.error_logs_path = os.path.join(self.log_path, 'errors.csv')

        print("There are {:d} test items\n".format(len(test_dataset)))

        self.save_opts()

    def test(self):
        """Testing the model on the Validation Set
        """

        self.set_eval()
        print("Testing")
        
        errors = {}
        for batch_idx, inputs in tqdm(enumerate(self.test_loader)):
            with torch.no_grad():

                # try:
                #     outputs = self.process_batch(inputs, mode="test")
                # except:
                #     print('Skipping')
                #     continue
                outputs = self.process_batch(inputs, errors, mode="test")
                if self.opts.save_imgs:
                    self.save_images(inputs, outputs)
                else:
                    self.compute_superglue_offset(inputs, outputs, errors)
                    self.compute_sift_offset(inputs, outputs, errors)
                    self.compute_loftr_offset(inputs, outputs, errors)
                    self.compute_pixel_errors(inputs, outputs, errors)
                    
                    try:
                        self.compute_of_errors(inputs, outputs, errors)
                    except AssertionError:
                        pass
                    
                    self.log_rotation_errors(inputs, outputs, errors)
                    self.log('test', inputs, outputs, errors)
                self.step += 1
        
        average_errors = {}
        
        for key, error in errors.items():
            average_errors[key] = sum(error)/len(error)

        df = pd.DataFrame.from_dict(average_errors)
        df.to_csv(self.error_logs_path)
        print("Error logs saved to {}".format(self.error_logs_path))

    def compute_loftr_offset(self, inputs, outputs, offsets, keypoint='loftr'):
        """Estimate average keypoint offset in the Y-axis
        """
        of_resize = self.opts.algolux_test_res_opt_flow

        offsets["loftr_offset/input"] = offsets.get("loftr_offset/input", []) + [torch.abs(outputs['loftr_matches/inp']['mkeypoints0'][:, 1] - outputs['loftr_matches/inp']['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]

        if self.opts.dataset in ['flyingthings', 'dstereo', 'sintel']:
            gt1 = F.interpolate(inputs['og_left'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(inputs['og_right'], scale_factor=(of_resize, of_resize))
        else:
            gt1 = F.interpolate(outputs['gt_left_rectf'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(outputs['gt_right_rectf'], scale_factor=(of_resize, of_resize))


        gt_matches = self.loftr(gt1, gt2)
        offsets["loftr_offset/gt"] = offsets.get("loftr_offset/gt", []) + [torch.abs(gt_matches['mkeypoints0'][:, 1] - gt_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
        outputs['loftr_matches/gt'] = gt_matches

        img1 = F.interpolate(outputs['pred_left_rectf'], scale_factor=(of_resize, of_resize))
        img2 = F.interpolate(outputs['pred_right_rectf'], scale_factor=(of_resize, of_resize))
        pred_matches = self.loftr(img1, img2)
        offsets["loftr_offset/pred"] = offsets.get("loftr_offset/pred", []) + [torch.abs(pred_matches['mkeypoints0'][:, 1] - pred_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
        outputs['loftr_matches/pred'] = pred_matches

    def compute_superglue_offset(self, inputs, outputs, offsets, keypoint='superglue'):
        """Estimate average keypoint offset in the Y-axis
        """
        of_resize = self.opts.algolux_test_res_opt_flow

        offsets["superglue_offset/input"] = offsets.get("superglue_offset/input", []) + [torch.abs(outputs['superglue_matches/inp']['mkeypoints0'][:, 1] - outputs['superglue_matches/inp']['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]

        if self.opts.dataset in ['flyingthings', 'dstereo', 'sintel']:
            gt1 = F.interpolate(inputs['og_left'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(inputs['og_right'], scale_factor=(of_resize, of_resize))
        else:
            gt1 = F.interpolate(outputs['gt_left_rectf'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(outputs['gt_right_rectf'], scale_factor=(of_resize, of_resize))

        gt_matches = self.superglue(gt1, gt2)
        offsets["superglue_offset/gt"] = offsets.get("superglue_offset/gt", []) + [torch.abs(gt_matches['mkeypoints0'][:, 1] - gt_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
        outputs['superglue_matches/gt'] = gt_matches

        '''
        img1 = F.interpolate(outputs['pred_left_rectf'], scale_factor=(of_resize, of_resize))
        img2 = F.interpolate(outputs['pred_right_rectf'], scale_factor=(of_resize, of_resize))
        pred_matches = self.superglue(img1, img2)
        offsets["superglue_offset/pred"] = offsets.get("superglue_offset/pred", []) + [torch.abs(pred_matches['mkeypoints0'][:, 1] - pred_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
        outputs['superglue_matches/pred'] = pred_matches
        '''

        inps_dict = {'pred': [outputs['pred_left_rectf'], outputs['pred_right_rectf']],
                     'sift': [outputs['sift_left_rectf'], outputs['sift_right_rectf']],
                     'superglue': [outputs['superglue_left_rectf'], outputs['superglue_right_rectf']],
                     'loftr': [outputs['loftr_left_rectf'], outputs['loftr_right_rectf']],
                     'dnet': [outputs['dnet_left_rectf'], outputs['dnet_right_rectf']],
                     'vit': [outputs['vit_left_rectf'], outputs['vit_right_rectf']],
                     'rpnet': [outputs['rpnet_left_rectf'], outputs['rpnet_right_rectf']],
                     'fepe': [outputs['fepe_left_rectf'], outputs['fepe_right_rectf']],
                     }

        for key, imgs in inps_dict.items():
            img1 = F.interpolate(imgs[0], scale_factor=(of_resize, of_resize))
            img2 = F.interpolate(imgs[1], scale_factor=(of_resize, of_resize))
            try:
                pred_matches = self.superglue(img1, img2)
                offsets[f"superglue_offset/{key}"] = offsets.get(f"superglue_offset/{key}", []) + [torch.abs(pred_matches['mkeypoints0'][:, 1] - pred_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
                outputs[f'superglue_matches/{key}'] = pred_matches
            except IndexError:
                pass


    def compute_sift_offset(self, inputs, outputs, offsets):
        """Estimate average keypoint offset in the Y-axis
        """
        of_resize = self.opts.algolux_test_res_opt_flow

        offsets["sift_offset/input"] = offsets.get("sift_offset/input", []) + [torch.abs(outputs['sift_matches/inp']['mkeypoints0'][:, 1] - outputs['sift_matches/inp']['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]

        if self.opts.dataset in ['flyingthings', 'dstereo', 'sintel']:
            gt1 = F.interpolate(inputs['og_left'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(inputs['og_right'], scale_factor=(of_resize, of_resize))
        else:
            gt1 = F.interpolate(outputs['gt_left_rectf'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(outputs['gt_right_rectf'], scale_factor=(of_resize, of_resize))

        gt_matches = find_sift_match(gt1, gt2)
        offsets["sift_offset/gt"] = offsets.get("sift_offset/gt", []) + [torch.abs(gt_matches['mkeypoints0'][:, 1] - gt_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
        outputs['sift_matches/gt'] = gt_matches

        '''
        img1 = F.interpolate(outputs['pred_left_rectf'], scale_factor=(of_resize, of_resize))
        img2 = F.interpolate(outputs['pred_right_rectf'], scale_factor=(of_resize, of_resize))
        pred_matches = find_sift_match(img1, img2)
        offsets["sift_offset/pred"] = offsets.get("sift_offset/pred", []) + [torch.abs(pred_matches['mkeypoints0'][:, 1] - pred_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
        outputs['sift_matches/pred'] = pred_matches
        '''
        if self.opts.dataset in ['carla', 'kitti']:

            inps_dict = {'pred': [outputs['pred_left_rectf'], outputs['pred_right_rectf']],
                        'sift': [outputs['sift_left_rectf'], outputs['sift_right_rectf']],
                        'superglue': [outputs['superglue_left_rectf'], outputs['superglue_right_rectf']],
                        'loftr': [outputs['loftr_left_rectf'], outputs['loftr_right_rectf']],
                        'dnet': [outputs['dnet_left_rectf'], outputs['dnet_right_rectf']],
                        'vit': [outputs['vit_left_rectf'], outputs['vit_right_rectf']],
                        'rpnet': [outputs['rpnet_left_rectf'], outputs['rpnet_right_rectf']],
                        'fepe': [outputs['fepe_left_rectf'], outputs['fepe_right_rectf']],
                        }

        elif self.opts.dataset in ['torc']:
            inps_dict = {'pred': [outputs['pred_left_rectf'], outputs['pred_right_rectf']],
                        'sift': [outputs['sift_left_rectf'], outputs['sift_right_rectf']],
                        'superglue': [outputs['superglue_left_rectf'], outputs['superglue_right_rectf']],
                        'loftr': [outputs['loftr_left_rectf'], outputs['loftr_right_rectf']],
                        'dnet': [outputs['dnet_left_rectf'], outputs['dnet_right_rectf']],
                        'vit': [outputs['vit_left_rectf'], outputs['vit_right_rectf']],
                        'rpnet': [outputs['rpnet_left_rectf'], outputs['rpnet_right_rectf']],
                        'fepe': [outputs['fepe_left_rectf'], outputs['fepe_right_rectf']],
                        }

        #print(inputs['img_id'])
        for key, imgs in inps_dict.items():
            img1 = F.interpolate(imgs[0], scale_factor=(of_resize, of_resize))
            img2 = F.interpolate(imgs[1], scale_factor=(of_resize, of_resize))
            #print(key)
            pred_matches = find_sift_match2(img1, img2)
            offsets[f"sift_offset/{key}"] = offsets.get(f"sift_offset/{key}", []) + [torch.abs(pred_matches['mkeypoints0'][:, 1] - pred_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]
            outputs[f'sift_matches/{key}'] = pred_matches


    def compute_of_errors(self, inputs, outputs, errors):
        of_height, of_width = self.opts.height, self.opts.width
        of_resize = self.opts.algolux_test_res_opt_flow

        if self.opts.dataset in ['flyingthings', 'dstereo', 'sintel']:
            gt1 = F.interpolate(inputs['og_left'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(inputs['og_right'], scale_factor=(of_resize, of_resize))
        else:
            gt1 = F.interpolate(outputs['gt_left_rectf'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(outputs['gt_right_rectf'], scale_factor=(of_resize, of_resize))

        inps_dict = {'pred': [outputs['pred_left_rectf'], outputs['pred_right_rectf']],
                     'gt': [gt1, gt2],
                     'input': [inputs['target_aug'], inputs['reference_aug']],
                     'sift': [outputs['sift_left_rectf'], outputs['sift_right_rectf']],
                     'superglue': [outputs['superglue_left_rectf'], outputs['superglue_right_rectf']],
                     'loftr': [outputs['loftr_left_rectf'], outputs['loftr_right_rectf']],
                     'dnet': [outputs['dnet_left_rectf'], outputs['dnet_right_rectf']],
                     'vit': [outputs['vit_left_rectf'], outputs['vit_right_rectf']],
                     'rpnet': [outputs['rpnet_left_rectf'], outputs['rpnet_right_rectf']],
                     'fepe': [outputs['fepe_left_rectf'], outputs['fepe_right_rectf']],
                     }
        
        for stage, img_list in inps_dict.items():

            if stage != 'input':
                img1 = F.interpolate(img_list[0], scale_factor=(of_resize, of_resize))
                img2 = F.interpolate(img_list[1], scale_factor=(of_resize, of_resize))
            else:
                img1, img2 = img_list[0], img_list[1]

            of_height, of_width = img1.shape[2], img1.shape[3]

            # calculate optical flow 
            flow_x, flow_y = self.opt_flow(img1.to(self.opt_flow_device), img2.to(self.opt_flow_device))

            if self.opts.use_sky_mask:
                flow_x, flow_y = flow_x[:, of_height//3:, :], flow_y[:, of_height//3:, :]

            flow_x = torch.nan_to_num(flow_x, nan = 0.0)

            assert not (torch.any(flow_x.isnan()) or torch.any(flow_y.isnan())), "NaN values detected in optical flow calculation"

            of_loss = torch.abs(flow_y).mean()
            
            assert not of_loss.isnan(), "optical flow loss is nan"

            errors['of_error/{}'.format(stage)] = errors.get('of_error/{}'.format(stage), []) + [of_loss]

            # store optical flow for logging
            outputs['flow_x/{}'.format(stage)] = flow_x.detach().cpu().numpy()
            outputs['flow_y/{}'.format(stage)] = flow_y.detach().cpu().numpy()

    def compute_pixel_errors(self, inputs, outputs, errors):

        of_resize = self.opts.algolux_test_res_opt_flow

        inps_dict = {'pred': [outputs['pred_left_rectf'], outputs['pred_right_rectf']],
                     'input': [inputs['full_left'], inputs['full_right']],
                     'sift': [outputs['sift_left_rectf'], outputs['sift_right_rectf']],
                     'superglue': [outputs['superglue_left_rectf'], outputs['superglue_right_rectf']],
                     'loftr': [outputs['loftr_left_rectf'], outputs['loftr_right_rectf']],
                     'dnet': [outputs['dnet_left_rectf'], outputs['dnet_right_rectf']],
                     'vit': [outputs['vit_left_rectf'], outputs['vit_right_rectf']],
                     'rpnet': [outputs['rpnet_left_rectf'], outputs['rpnet_right_rectf']],
                     'fepe': [outputs['fepe_left_rectf'], outputs['fepe_right_rectf']],
                     }
        
        if self.opts.dataset in ['flyingthings', 'dstereo', 'sintel']:
            gt_img1 = inputs['og_left']
            gt_img2 = inputs['og_right']
        else:
            gt_img1 = outputs['gt_left_rectf']
            gt_img2 = outputs['gt_right_rectf']
        
        for stage, img_list in inps_dict.items():
            pixel_loss = (torch.abs(img_list[0] - gt_img1) + torch.abs(img_list[1] - gt_img2)).mean()
            errors['pixel_error/{}'.format(stage)] = errors.get('pixel_error/{}'.format(stage), []) + [pixel_loss]

    def process_batch(self, inputs, errors, mode="train"):
        """Pass a minibatch through the network and generate images and losses
        """
        outputs = {}
        for key, ipt in inputs.items():
            if not key in ["img_id",'dataset']:
                if self.opts.use_dp:
                    inputs[key] = ipt.cuda(non_blocking=True)
                else:
                    inputs[key] = ipt.to(self.device)

        start = time.time()
        if self.opts.use_opt_flow_input:
            flow_x, flow_y = self.opt_flow(
                inputs['target_aug'].to(self.opt_flow_device), 
                inputs['reference_aug'].to(self.opt_flow_device)
                )
            inputs['init_flow_x'] = torch.nan_to_num(flow_x.detach().to(self.device), nan = 0.0).unsqueeze(dim=1)
            inputs['init_flow_y'] = torch.nan_to_num(flow_y.detach().to(self.device), nan = 0.0).unsqueeze(dim=1)

        if self.opts.model_name == 'posecnn':
            pose_inputs = [inputs['target_aug'], inputs['reference_aug']]
            pose_inputs = torch.cat(pose_inputs, 1)
            rotation, translation = self.models["pose"](pose_inputs)
            outputs["rotation"] = rotation
            outputs["translation"] = translation

        elif self.opts.model_name in ["siamese", "siamese_deep", "hsm"]:
            if self.opts.use_opt_flow_input:
                rotation, _ = self.models["pose"](
                    torch.cat((inputs['target_aug'], inputs['init_flow_x'], inputs['init_flow_y']), dim=1), 
                    torch.cat((inputs['reference_aug'], inputs['init_flow_x'], inputs['init_flow_y']), dim=1)
                    )
            elif self.opts.use_seq_hsm:
                rotation, cost_vol3_dict, pred_rots = self.models["pose"](inputs['target_aug'], inputs['reference_aug'], 
                                                                          inputs['translation'].unsqueeze(-1), 
                                                                          inputs["intrns_left"], inputs['intrns_right'],
                                                                          inputs['dist_coeff1'], inputs['dist_coeff2'],
                                                                          inputs['full_height'][0], inputs['full_width'][0])

                outputs['cost_vol3_dict'] = cost_vol3_dict
                outputs['predicted_rotations'] = pred_rots
            else:
                rotation, cost_vol3_dict = self.models["pose"](
                    inputs['target_aug'], inputs['reference_aug'])
                
                outputs['cost_vol3_dict'] = cost_vol3_dict
        elif self.opts.model_name in ['gmflow']:
            rotation = self.models["pose"](inputs['target_aug'], inputs['reference_aug'],
                                           inputs['translation'].unsqueeze(-1), 
                                           inputs["intrns_left"], inputs['intrns_right'],
                                           inputs['dist_coeff1'], inputs['dist_coeff2'],
                                           inputs['full_height'][0], inputs['full_width'][0])
            
            if self.opts.pose_repr in ['use_perturb', 'use_identity', 'use_yaw_perturb', 'use_yaw']:
                outputs["rotation"] = rotation.reshape(-1, 3, 3) + inputs['rotation_perturb']
            elif self.opts.pose_repr == ['use_perturb_6D', 'use_perturb_quat']:
                outputs["rotation"] = rotation + inputs['rotation_perturb']
            elif self.opts.pose_repr == "use_rectify":
                outputs["R1"] = rotation[:, :6]
                outputs["R2"] = rotation[:, 6:]
                outputs['rotation'] = rotation[:, 6:]
            else:
                outputs['rotation'] = rotation

        model_time = time.time() - start

        start = time.time()
        input_matches = self.superglue(inputs['target_aug'], inputs['reference_aug'])
        outputs['superglue_matches/inp'] = input_matches
        
        superglue_rotation = magsac_pose_estimation(input_matches['mkeypoints0'], input_matches['mkeypoints1'], input_matches['match_confidence'],
                                                    inputs['intrns_left'].to(self.device), inputs['intrns_right'].to(self.device),hw=inputs['target_aug'].shape[2:],
                                                    filter_conf=0.9)
        '''
        
        superglue_rotation, _, _ = estimate_pose_kornia(input_matches['mkeypoints0'], input_matches['mkeypoints1'], 
                                                   inputs['intrns_left'].to(self.device), inputs['intrns_right'].to(self.device), 
                                                   input_matches['match_confidence'],
                                                   filter_conf=0.9)
        '''

        if self.opts.pose_repr == 'use_9D':
            outputs['superglue_rotation'] = superglue_rotation
        elif self.opts.pose_repr == 'use_6D':
            outputs['superglue_rotation'] = superglue_rotation.view(-1, 9)[:, :6]

        superglue_time = time.time() - start

        start = time.time()
        input_matches, sift_rotation = opencv_pose_estimation(inputs['target_aug'], inputs['reference_aug'],
                                                              inputs['intrns_left'], inputs['intrns_right'])
        outputs['sift_matches/inp'] = input_matches

        if self.opts.pose_repr == 'use_9D':
            outputs['sift_rotation'] = sift_rotation
        elif self.opts.pose_repr == 'use_6D':
            outputs['sift_rotation'] = sift_rotation.view(-1, 9)[:, :6]

        sift_time = time.time() - start

        start = time.time()
        input_matches = self.loftr(inputs['target_aug'], inputs['reference_aug'])
        outputs['loftr_matches/inp'] = input_matches
        
        loftr_rotation = magsac_pose_estimation(input_matches['mkeypoints0'], input_matches['mkeypoints1'], input_matches['match_confidence'],
                                                inputs['intrns_left'].to(self.device), inputs['intrns_right'].to(self.device), hw=inputs['target_aug'].shape[2:],
                                                filter_conf=0.9)
        '''
        loftr_rotation, _, _ = estimate_pose_kornia(input_matches['mkeypoints0'], input_matches['mkeypoints1'], 
                                                   inputs['intrns_left'].to(self.device), inputs['intrns_right'].to(self.device), 
                                                   input_matches['match_confidence'],
                                                   filter_conf=0.9)
        '''

        if self.opts.pose_repr == 'use_9D':
            outputs['loftr_rotation'] = loftr_rotation
        elif self.opts.pose_repr == 'use_6D':
            outputs['loftr_rotation'] = loftr_rotation.view(-1, 9)[:, :6]
            
        loftr_time = time.time() - start

        time_dict = {'pred': model_time,
                     'superglue': superglue_time,
                     'sift': sift_time,
                     'loftr': loftr_time,
                     }

        for key, val in time_dict.items():
            errors[f"inference_time/{key}"] = errors.get(f"inference_time/{key}", []) + [val]

        self.generate_rectf_imgs(inputs, outputs)

        return outputs

    def generate_rectf_imgs(self, inputs, outputs):
        pred_rotation = outputs["rotation"]
        gt_rotation = inputs["rotation"]
        dnet_rmat = inputs["R_dnet"]
        vit_rmat = inputs["R_vit"]
        rpnet_rmat = inputs["R_rpnet"]
        fepe_rmat = inputs["R_fepe"]

        if self.opts.pose_repr in ['use_6D', 'use_perturb_6D']:
            pred_rmat = gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:])
            gt_rmat = gram_schmidt(gt_rotation[:, :3], gt_rotation[:, 3:])
            superglue_rmat = gram_schmidt(outputs['superglue_rotation'][:, :3], outputs['superglue_rotation'][:, 3:])
            sift_rmat = gram_schmidt(outputs['sift_rotation'][:, :3], outputs['sift_rotation'][:, 3:])
            loftr_rmat = gram_schmidt(outputs['loftr_rotation'][:, :3], outputs['loftr_rotation'][:, 3:])
        elif self.opts.pose_repr in ['use_9D', 'use_perturb', 'use_yaw_perturb', 'use_identity', 'use_yaw']:
            pred_rmat = svd_orthogonalize(pred_rotation.reshape(-1, 3, 3))
            gt_rmat = gt_rotation
            superglue_rmat = outputs['superglue_rotation']
            sift_rmat = outputs['sift_rotation']
            loftr_rmat = outputs['loftr_rotation']
        elif self.opts.pose_repr in ['use_quat', 'use_perturb_quat']:
            pred_rmat = quaternion_to_matrix(pred_rotation)
            gt_rmat = quaternion_to_matrix(gt_rotation)
        elif self.opts.pose_repr == "use_rectify":
            pred_r1, pred_r2 = gram_schmidt(outputs["R1"][:, :3], outputs["R1"][:, 3:]), gram_schmidt(outputs["R2"][:, :3], outputs["R2"][:, 3:])
            gt_r1, gt_r2 = inputs["R1"], inputs["R2"]
            P1, P2 = inputs["P1"], inputs["P2"]
        else:
            pred_rmat = pred_rotation
            gt_rmat = gt_rotation

        translation = inputs['translation'].unsqueeze(-1)

        intrinsics_left, intrinsics_right = inputs["intrns_left"], inputs['intrns_right']

        dist1 = inputs['dist_coeff1']
        dist2 = inputs['dist_coeff2']

        if self.opts.pose_repr == "use_rectify":
            outputs['pred_left_rectf'], outputs['pred_right_rectf'] = \
                undistortrectify(inputs['full_left'], intrinsics_left, dist1, pred_r1, P1[...,:3]), undistortrectify(inputs['full_right'], intrinsics_right, dist2, pred_r2, P2[...,:3])

            outputs['gt_left_rectf'], outputs['gt_right_rectf'] = \
                undistortrectify(inputs['full_left'], intrinsics_left, dist1, gt_r1, P1[...,:3]), undistortrectify(inputs['full_right'], intrinsics_right, dist2, gt_r2, P2[...,:3])
        else:
            outputs['pred_left_rectf'], outputs['pred_right_rectf'], _ = \
                calibrate_stereo_pair_torch(
                    inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, pred_rmat.double(), translation.double(),
                    inputs['full_height'][0], inputs['full_width'][0])
            
            outputs['gt_left_rectf'], outputs['gt_right_rectf'], _ = \
                calibrate_stereo_pair_torch(
                    inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, gt_rmat.double(), translation.double(),
                    inputs['full_height'][0], inputs['full_width'][0])

        outputs['superglue_left_rectf'], outputs['superglue_right_rectf'], _ = \
            calibrate_stereo_pair_torch(
                inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, superglue_rmat.double(), translation.double(),
                inputs['full_height'][0], inputs['full_width'][0])

        outputs['sift_left_rectf'], outputs['sift_right_rectf'], _ = \
            calibrate_stereo_pair_torch(
                inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, sift_rmat.double(), translation.double(),
                inputs['full_height'][0], inputs['full_width'][0])

        outputs['loftr_left_rectf'], outputs['loftr_right_rectf'], _ = \
            calibrate_stereo_pair_torch(
                inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, loftr_rmat.double(), translation.double(),
                inputs['full_height'][0], inputs['full_width'][0])

        outputs['dnet_left_rectf'], outputs['dnet_right_rectf'], _ = \
            calibrate_stereo_pair_torch(
                inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, dnet_rmat.double(), translation.double(),
                inputs['full_height'][0], inputs['full_width'][0])

        outputs['vit_left_rectf'], outputs['vit_right_rectf'], _ = \
            calibrate_stereo_pair_torch(
                inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, vit_rmat.double(), translation.double(),
                inputs['full_height'][0], inputs['full_width'][0])

        outputs['rpnet_left_rectf'], outputs['rpnet_right_rectf'], _ = \
            calibrate_stereo_pair_torch(
                inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, rpnet_rmat.double(), translation.double(),
                inputs['full_height'][0], inputs['full_width'][0])

        outputs['fepe_left_rectf'], outputs['fepe_right_rectf'], _ = \
            calibrate_stereo_pair_torch(
                inputs['full_left'], inputs['full_right'], intrinsics_left, intrinsics_right, dist1, dist2, fepe_rmat.double(), translation.double(),
                inputs['full_height'][0], inputs['full_width'][0])

    def compute_rotation_errors(self, pred_rotation, gt_axisangles, skip_conversion=False):

        if not skip_conversion:
            if self.opts.pose_repr == 'use_3D':
                pred_axisangles = pred_rotation.numpy().reshape(-1, 3)
            elif self.opts.pose_repr in ['use_quat', 'use_perturb_quat']:
                pred_rotation = pred_rotation.numpy().reshape(-1, 4)
                pred_axisangles = Rotation.from_quat(
                    pred_rotation).as_rotvec(degrees=True).astype(np.float32)
            elif self.opts.pose_repr == 'use_6D':
                pred_rotation = pred_rotation.reshape(-1, 6)
                pred_axisangles = Rotation.from_matrix(gram_schmidt(
                    pred_rotation[:, :3], pred_rotation[:, 3:]).numpy()).as_rotvec(degrees=True).astype(np.float32)
            elif self.opts.pose_repr == 'use_9D':
                pred_rotation = pred_rotation.reshape(-1, 3, 3)
                pred_axisangles = Rotation.from_matrix(svd_orthogonalize(pred_rotation).numpy()).as_rotvec(degrees=True).astype(np.float32)
            elif self.opts.pose_repr in ['use_perturb', 'use_yaw_perturb', 'use_identity', 'use_yaw']:
                pred_rotation = pred_rotation
                pred_axisangles = Rotation.from_matrix(svd_orthogonalize(pred_rotation).numpy()).as_rotvec(degrees=True).astype(np.float32)
            elif self.opts.pose_repr == 'use_perturb_6D':
                pred_rotation = pred_rotation.reshape(-1, 6)
                pred_axisangles = Rotation.from_matrix(gram_schmidt(
                    pred_rotation[:, :3], pred_rotation[:, 3:]).numpy()).as_rotvec(degrees=True).astype(np.float32)
        else:
            pred_axisangles = pred_rotation

        delta = np.absolute(gt_axisangles - pred_axisangles)

        return delta

    def log_rotation_errors(self, inputs, outputs, errors):
        gt_axisangles = inputs['axisangles'].cpu().numpy()
        rotation_dict = {"pred": outputs["rotation"].detach().cpu(), 
                         'superglue': outputs["superglue_rotation"].detach().cpu(),
                         'sift': outputs["sift_rotation"].detach().cpu(),
                         'loftr': outputs["loftr_rotation"].detach().cpu(),
                         'dnet': inputs["axisangles_dnet"].detach().cpu().numpy(),
                         'vit': inputs["axisangles_vit"].detach().cpu().numpy(),
                         'rpnet': inputs["axisangles_rpnet"].detach().cpu().numpy(),
                         'fepe': inputs["axisangles_fepe"].detach().cpu().numpy(),
                         }

        for stage, rotation in rotation_dict.items():
            if stage in ["dnet", 'vit', 'rpnet', 'fepe']:
                delta = self.compute_rotation_errors(rotation, gt_axisangles, skip_conversion=True)
            else:
                delta = self.compute_rotation_errors(rotation, gt_axisangles)

            errors['{}(degrees)/delta_x'.format(stage)] = errors.get('{}(degrees)/delta_x'.format(stage), []) + [delta[:, 0].mean()]
            errors['{}(degrees)/delta_y'.format(stage)] = errors.get('{}(degrees)/delta_y'.format(stage), []) + [delta[:, 1].mean()]
            errors['{}(degrees)/delta_z'.format(stage)] = errors.get('{}(degrees)/delta_z'.format(stage), []) + [delta[:, 2].mean()]

    def log(self, mode, inputs, outputs, errors):

        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        if "siamese" in self.opts.model_name:
            for i, (layer, param) in enumerate(self.models["pose"].named_parameters()):
                if param.requires_grad:
                    writer.add_histogram("{}_grad".format(layer), param.grad, self.step) 
                if i > 4:
                    break

        # Generate rectified images from both ground truth and predicted poses
        pred_rotation = outputs["rotation"].detach().cpu()
        sift_rotation = outputs["sift_rotation"].detach().cpu()
        superglue_rotation = outputs["superglue_rotation"].detach().cpu()
        loftr_rotation = outputs["loftr_rotation"].detach().cpu()
        gt_rotation = inputs["rotation"].detach().cpu()
        
        if self.opts.pose_repr in ['use_perturb', 'use_perturb_quat', 'use_perturb_6D', 'use_identity','use_yaw_perturb']:
            init_rotation = inputs["rotation_perturb"].detach().cpu()

        if self.opts.pose_repr in ['use_6D', 'use_perturb_6D']:
            pred_rmat = Rotation.from_matrix(gram_schmidt(
                pred_rotation[:, :3], pred_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            gt_rmat = Rotation.from_matrix(gram_schmidt(
                gt_rotation[:, :3], gt_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            superglue_rmat = Rotation.from_matrix(gram_schmidt(
                superglue_rotation[:, :3], superglue_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            sift_rmat = Rotation.from_matrix(gram_schmidt(
                sift_rotation[:, :3], sift_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            loftr_rmat = Rotation.from_matrix(gram_schmidt(
                loftr_rotation[:, :3], loftr_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            if self.opts.pose_repr == 'use_perturb_6D':
                init_rmat = Rotation.from_matrix(gram_schmidt(
                    init_rotation[:, :3], init_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)

        elif self.opts.pose_repr in ['use_9D', 'use_perturb', 'use_identity', 'use_yaw_perturb', 'use_yaw']:
            pred_rotation = pred_rotation.reshape(-1, 3, 3)
            pred_rmat = svd_orthogonalize(pred_rotation).numpy().astype(np.float32)
            gt_rmat = gt_rotation.numpy().astype(np.float32)
            superglue_rmat = superglue_rotation.numpy().astype(np.float32)
            sift_rmat = sift_rotation.numpy().astype(np.float32)
            loftr_rmat = loftr_rotation.numpy().astype(np.float32)

            if self.opts.pose_repr in ['use_perturb', 'use_identity', 'use_yaw_perturb']:
                init_rmat = init_rotation.numpy().astype(np.float32)        

        elif self.opts.pose_repr in ['use_quat', 'use_perturb_quat']:
            pred_rotation = pred_rotation.numpy().reshape(-1, 4)
            pred_rmat = Rotation.from_quat(pred_rotation).as_matrix().astype(np.float32)
            gt_rotation = gt_rotation.numpy().reshape(-1, 4)
            gt_rmat = Rotation.from_quat(gt_rotation).as_matrix().astype(np.float32)
            if self.opts.pose_repr == 'use_perturb_quat':
                init_rotation = init_rotation.numpy().reshape(-1, 4)
                init_rmat = Rotation.from_quat(init_rotation).as_matrix().astype(np.float32)

        elif self.opts.pose_repr == "use_rectify":
            pred_r1, pred_r2 = gram_schmidt(outputs["R1"][:, :3], outputs["R1"][:, 3:]).detach().cpu().numpy(), gram_schmidt(outputs["R2"][:, :3], outputs["R2"][:, 3:]).detach().cpu().numpy()
            gt_r1, gt_r2 = inputs["R1"].cpu().numpy(), inputs["R2"].cpu().numpy()
            P1, P2 = inputs["P1"].cpu().numpy(), inputs["P2"].cpu().numpy()

        dnet_rmat = inputs["R_dnet"].detach().cpu().numpy()
        vit_rmat = inputs["R_vit"].detach().cpu().numpy()
        rpnet_rmat = inputs["R_rpnet"].detach().cpu().numpy()
        fepe_rmat = inputs["R_fepe"].detach().cpu().numpy()

        translation = inputs['translation'].detach().cpu().numpy()

        left_imgs = inputs['full_left'].permute(0, 2, 3, 1).cpu().numpy()
        right_imgs = inputs['full_right'].permute(0, 2, 3, 1).cpu().numpy()

        for j in range(min(4, self.opts.batch_size)):
            unrect_left_img = (left_imgs[j, ...] *255.0).astype(np.uint8)
            unrect_right_img = (right_imgs[j, ...] *255.0).astype(np.uint8)

            height,width,_ = unrect_left_img.shape

            if self.opts.pose_repr != "use_rectify":
                R_pred = pred_rmat[j,...]
                R_sglue = superglue_rmat[j, ...]
                R_sift = sift_rmat[j, ...]
                R_loftr = loftr_rmat[j, ...]
                R_gt = gt_rmat[j,...]
                R_dnet = dnet_rmat[j, ...]
                R_vit = vit_rmat[j, ...]
                R_rpnet = rpnet_rmat[j, ...]
                R_fepe = fepe_rmat[j, ...]
            
            t = translation[j,...]

            K_left, K_right = inputs["intrns_left"][j].cpu().numpy(), inputs['intrns_right'][j].cpu().numpy()

            dist1 = inputs['dist_coeff1'][j].cpu().numpy()
            dist2 = inputs['dist_coeff2'][j].cpu().numpy()

            pred_img_left_rect, pred_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_pred.astype(np.float64), t.astype(np.float64), height, width)

            superglue_img_left_rect, superglue_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_sglue.astype(np.float64), t.astype(np.float64), height, width)

            sift_img_left_rect, sift_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_sift.astype(np.float64), t.astype(np.float64), height, width)

            loftr_img_left_rect, loftr_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_loftr.astype(np.float64), t.astype(np.float64), height, width)

            dnet_img_left_rect, dnet_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_dnet.astype(np.float64), t.astype(np.float64), height, width)

            vit_img_left_rect, vit_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_vit.astype(np.float64), t.astype(np.float64), height, width)

            rpnet_img_left_rect, rpnet_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_rpnet.astype(np.float64), t.astype(np.float64), height, width)

            fepe_img_left_rect, fepe_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_fepe.astype(np.float64), t.astype(np.float64), height, width)

            if self.opts.dataset in ['flyingthings', 'dstereo', 'sintel']:
                og_lefts = inputs['og_left'].permute(0, 2, 3, 1).cpu().numpy()
                og_rights = inputs['og_right'].permute(0, 2, 3, 1).cpu().numpy()

                gt_img_left_rect = (og_lefts[j, ...].copy() *255.0).astype(np.uint8)
                gt_img_right_rect = (og_rights[j, ...].copy() *255.0).astype(np.uint8)

            else:
                gt_img_left_rect, gt_img_right_rect, _ = calibrate_stereo_pair(
                    unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_gt.astype(np.float64), t.astype(np.float64), height, width)
                
            pred_ep_image_pair = draw_epipolar_lines(pred_img_left_rect, pred_img_right_rect, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
            superglue_ep_image_pair = draw_epipolar_lines(superglue_img_left_rect, superglue_img_right_rect, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
            sift_ep_image_pair = draw_epipolar_lines(sift_img_left_rect, sift_img_right_rect, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
            loftr_ep_image_pair = draw_epipolar_lines(loftr_img_left_rect, loftr_img_right_rect, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
            gt_ep_image_pair = draw_epipolar_lines(gt_img_left_rect, gt_img_right_rect, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
            input_ep_image_pair = draw_epipolar_lines(unrect_left_img.copy(), unrect_right_img.copy(), num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])

            pred_image_overlay = cv2.addWeighted(pred_img_left_rect, 0.5, 
                                                 pred_img_right_rect, 0.5, 0)

            sglue_image_overlay = cv2.addWeighted(superglue_img_left_rect, 0.5, 
                                                 superglue_img_right_rect, 0.5, 0)

            sift_image_overlay = cv2.addWeighted(sift_img_left_rect, 0.5, 
                                                 sift_img_right_rect, 0.5, 0)

            loftr_image_overlay = cv2.addWeighted(loftr_img_left_rect, 0.5, 
                                                 loftr_img_right_rect, 0.5, 0)

            gt_image_overlay = cv2.addWeighted(gt_img_left_rect, 0.5, 
                                               gt_img_right_rect, 0.5, 0)

            input_image_overlay = cv2.addWeighted(unrect_left_img, 0.5, 
                                                  unrect_right_img, 0.5, 0)

            dnet_image_overlay = cv2.addWeighted(dnet_img_left_rect, 0.5, 
                                                dnet_img_right_rect, 0.5, 0)

            vit_image_overlay = cv2.addWeighted(vit_img_left_rect, 0.5, 
                                                vit_img_right_rect, 0.5, 0)

            rpnet_image_overlay = cv2.addWeighted(rpnet_img_left_rect, 0.5, 
                                                rpnet_img_right_rect, 0.5, 0)

            fepe_image_overlay = cv2.addWeighted(fepe_img_left_rect, 0.5, 
                                                fepe_img_right_rect, 0.5, 0)

            scale_percent = 50 # percent of original size
            width = int(pred_ep_image_pair.shape[1] * scale_percent / 100)
            height = int(pred_ep_image_pair.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            pred_ep_image_pair = cv2.resize(pred_ep_image_pair, dim, interpolation = cv2.INTER_AREA)
            sift_ep_image_pair = cv2.resize(sift_ep_image_pair, dim, interpolation = cv2.INTER_AREA)
            superglue_ep_image_pair = cv2.resize(superglue_ep_image_pair, dim, interpolation = cv2.INTER_AREA)
            loftr_ep_image_pair = cv2.resize(loftr_ep_image_pair, dim, interpolation = cv2.INTER_AREA)
            gt_ep_image_pair = cv2.resize(gt_ep_image_pair, dim, interpolation = cv2.INTER_AREA)

            writer.add_image("input_epipolar_images/{}".format(j), input_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("pred_epipolar_images/{}".format(j), pred_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("superglue_epipolar_images/{}".format(j), superglue_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("sift_epipolar_images/{}".format(j), sift_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("loftr_epipolar_images/{}".format(j), loftr_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("gt_epipolar_images/{}".format(j), gt_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("input_overlay/{}".format(j), input_image_overlay, self.step, dataformats='HWC')
            writer.add_image("pred_overlay/{}".format(j), pred_image_overlay, self.step, dataformats='HWC')
            writer.add_image("gt_overlay/{}".format(j), gt_image_overlay, self.step, dataformats='HWC')
            writer.add_image("superglue_overlay/{}".format(j), sglue_image_overlay, self.step, dataformats='HWC')
            writer.add_image("sift_overlay/{}".format(j), sift_image_overlay, self.step, dataformats='HWC')
            writer.add_image("loftr_overlay/{}".format(j), loftr_image_overlay, self.step, dataformats='HWC')
            writer.add_image("dnet_overlay/{}".format(j), dnet_image_overlay, self.step, dataformats='HWC')
            writer.add_image("vit_overlay/{}".format(j), vit_image_overlay, self.step, dataformats='HWC')
            writer.add_image("rpnet_overlay/{}".format(j), rpnet_image_overlay, self.step, dataformats='HWC')
            writer.add_image("fepe_overlay/{}".format(j), fepe_image_overlay, self.step, dataformats='HWC')
            
            # optical flow visualization
            for key in ["superglue", "sift", "pred", "loftr"]:
                if 'flow_x/{}'.format(key) in outputs.keys():
                    flow_x = np.clip(outputs['flow_x/{}'.format(key)][j],0.0,255.0)
                    flow_y = np.clip(outputs['flow_y/{}'.format(key)][j],0.0,255.0)
                    f = np.concatenate((flow_x[...,None],flow_y[...,None]),axis=-1)
                    opt_flow_img = flow_to_image(f)
                    writer.add_image("optical_flow/{}_{}".format(key, j), opt_flow_img, self.step, dataformats='HWC')
            
            for key, offsets in errors.items():
                writer.add_scalar("{}".format(key), sum(offsets)/len(offsets), self.step)

            '''
            for key in ["sift", "superglue", "loftr"]:
                of_resize = self.opts.algolux_test_res_opt_flow
                color = cm.jet(outputs['{}_matches/pred'.format(key)]['match_confidence'].detach().cpu().numpy())
                kps0 = outputs['{}_matches/pred'.format(key)]['mkeypoints0'].detach().cpu().numpy() / of_resize
                kps1 = outputs['{}_matches/pred'.format(key)]['mkeypoints1'].detach().cpu().numpy() / of_resize

                matching_plot = make_matching_plot_fast(pred_img_left_rect, pred_img_right_rect,
                                                        kps0, kps1, 
                                                        kps0, kps1,
                                                        color)
                
                writer.add_image("{}_matches/{}".format(key, j), matching_plot, self.step, dataformats='HWC')
            '''

    def save_images(self, inputs, outputs):
        # Generate rectified images from both ground truth and predicted poses
        pred_rotation = outputs["rotation"].detach().cpu()
        sift_rotation = outputs["sift_rotation"].detach().cpu()
        superglue_rotation = outputs["superglue_rotation"].detach().cpu()
        loftr_rotation = outputs["loftr_rotation"].detach().cpu()
        gt_rotation = inputs["rotation"].detach().cpu()
        if self.opts.pose_repr in ['use_perturb', 'use_perturb_quat', 'use_perturb_6D', 'use_identity','use_yaw_perturb']:
            init_rotation = inputs["rotation_perturb"].detach().cpu()

        if self.opts.pose_repr in ['use_6D', 'use_perturb_6D']:
            pred_rmat = Rotation.from_matrix(gram_schmidt(
                pred_rotation[:, :3], pred_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            gt_rmat = Rotation.from_matrix(gram_schmidt(
                gt_rotation[:, :3], gt_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            superglue_rmat = Rotation.from_matrix(gram_schmidt(
                superglue_rotation[:, :3], superglue_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            sift_rmat = Rotation.from_matrix(gram_schmidt(
                sift_rotation[:, :3], sift_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            loftr_rmat = Rotation.from_matrix(gram_schmidt(
                loftr_rotation[:, :3], loftr_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            if self.opts.pose_repr == 'use_perturb_6D':
                init_rmat = Rotation.from_matrix(gram_schmidt(
                    init_rotation[:, :3], init_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)

        elif self.opts.pose_repr in ['use_9D', 'use_perturb', 'use_identity', 'use_yaw_perturb', 'use_yaw']:
            pred_rotation = pred_rotation.reshape(-1, 3, 3)
            pred_rmat = svd_orthogonalize(pred_rotation).numpy().astype(np.float32)
            gt_rmat = gt_rotation.numpy().astype(np.float32)
            superglue_rmat = superglue_rotation.numpy().astype(np.float32)
            sift_rmat = sift_rotation.numpy().astype(np.float32)
            loftr_rmat = loftr_rotation.numpy().astype(np.float32)

            if self.opts.pose_repr in ['use_perturb', 'use_identity', 'use_yaw_perturb']:
                init_rmat = init_rotation.numpy().astype(np.float32)        

        dnet_rmat = inputs["R_dnet"].detach().cpu().numpy()
        vit_rmat = inputs["R_vit"].detach().cpu().numpy()
        rpnet_rmat = inputs["R_rpnet"].detach().cpu().numpy()
        fepe_rmat = inputs["R_fepe"].detach().cpu().numpy()

        translation = inputs['translation'].detach().cpu().numpy()

        left_imgs = inputs['full_left'].permute(0, 2, 3, 1).cpu().numpy()
        right_imgs = inputs['full_right'].permute(0, 2, 3, 1).cpu().numpy()

        for j in range(min(4, self.opts.batch_size)):
            unrect_left_img = (left_imgs[j, ...] *255.0).astype(np.uint8)
            unrect_right_img = (right_imgs[j, ...] *255.0).astype(np.uint8)

            height,width,_ = unrect_left_img.shape

            if self.opts.pose_repr != "use_rectify":
                R_pred = pred_rmat[j,...]
                R_sglue = superglue_rmat[j, ...]
                R_sift = sift_rmat[j, ...]
                R_loftr = loftr_rmat[j, ...]
                R_gt = gt_rmat[j,...]
                R_dnet = dnet_rmat[j, ...]
                R_vit = vit_rmat[j, ...]
                R_rpnet = rpnet_rmat[j, ...]
                R_fepe = fepe_rmat[j, ...]
            
            t = translation[j,...]

            K_left, K_right = inputs["intrns_left"][j].cpu().numpy(), inputs['intrns_right'][j].cpu().numpy()

            dist1 = inputs['dist_coeff1'][j].cpu().numpy()
            dist2 = inputs['dist_coeff2'][j].cpu().numpy()

            pred_img_left_rect, pred_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_pred.astype(np.float64), t.astype(np.float64), height, width)

            superglue_img_left_rect, superglue_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_sglue.astype(np.float64), t.astype(np.float64), height, width)

            sift_img_left_rect, sift_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_sift.astype(np.float64), t.astype(np.float64), height, width)

            loftr_img_left_rect, loftr_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_loftr.astype(np.float64), t.astype(np.float64), height, width)

            gt_img_left_rect, gt_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_gt.astype(np.float64), t.astype(np.float64), height, width)

            dnet_img_left_rect, dnet_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_dnet.astype(np.float64), t.astype(np.float64), height, width)

            vit_img_left_rect, vit_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_vit.astype(np.float64), t.astype(np.float64), height, width)

            rpnet_img_left_rect, rpnet_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_rpnet.astype(np.float64), t.astype(np.float64), height, width)

            fepe_img_left_rect, fepe_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_fepe.astype(np.float64), t.astype(np.float64), height, width)

            images_dict = {'pred': [pred_img_left_rect, pred_img_right_rect, R_pred],
                           'gt': [gt_img_left_rect, gt_img_right_rect, R_gt],
                           'inp': [unrect_left_img, unrect_right_img, np.eye(3)],
                           'sift': [sift_img_left_rect, sift_img_right_rect, R_sift],
                           'superglue': [superglue_img_left_rect, superglue_img_right_rect, R_sglue],
                           'loftr': [loftr_img_left_rect, loftr_img_right_rect, R_loftr],
                            'dnet': [dnet_img_left_rect, dnet_img_right_rect, R_dnet],
                            'vit': [vit_img_left_rect, vit_img_right_rect, R_vit],
                            'rpnet': [rpnet_img_left_rect, rpnet_img_right_rect, R_rpnet],
                            'fepe': [fepe_img_left_rect, fepe_img_right_rect, R_fepe],
                            }
            
            for method, imgs in images_dict.items():
                save_dir = os.path.join(self.log_path, 'saved_imgs', method)
                if not os.path.isdir(save_dir):
                    os.makedirs(os.path.join(save_dir, 'left'))
                    os.makedirs(os.path.join(save_dir, 'right'))
                    os.makedirs(os.path.join(save_dir, 'extrinscs'))
                if not os.path.isdir(os.path.join(save_dir, 'intrinsics', 'left')):
                    os.makedirs(os.path.join(save_dir, 'intrinsics', 'left'))
                    os.makedirs(os.path.join(save_dir, 'intrinsics', 'right'))

                id = inputs['img_id'][0]
                cv2.imwrite(os.path.join(save_dir, 'left', f'{id}.png'), imgs[0][:, :, ::-1])
                cv2.imwrite(os.path.join(save_dir, 'right', f'{id}.png'), imgs[1][:, :, ::-1])
                T = np.eye(4)
                T[:3, :3] = imgs[-1]
                T[:3, 3] = t
                np.save(os.path.join(save_dir, 'extrinscs', f'{id}.npy'), T)
                np.save(os.path.join(save_dir, 'intrinsics', 'left', f'{id}.npy'), K_left.astype(np.float64))
                np.save(os.path.join(save_dir, 'intrinsics', 'right', f'{id}.npy'), K_right.astype(np.float64))