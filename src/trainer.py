from cmath import nan
from gettext import translation
import os
import time
from matplotlib.text import get_rotation
import numpy as np
import cv2
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

import model
import dataset
from utility.data_utils import calibrate_stereo_pair,calibrate_stereo_pair_torch, draw_epipolar_lines, gram_schmidt, svd_orthogonalize, undistortrectify, undistortrectify_cv2
from utility.segmentation import DLV3P
from utility.opticalflow import OpticalFlow, TorchvisionRAFT
from utility.log_utils import flow_to_image

torch.autograd.set_detect_anomaly(True)

class Trainer(BaseTrainer):

    def __init__(self, options) -> None:
        super(Trainer, self).__init__()
        self.opts = options
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
            self.models['pose'] = model.GMFlow(n_out=self.num_parameters[self.opts.pose_repr], dataset=self.opts.dataset, ablate_init=self.opts.ablate_init, ablate_transformer=self.opts.ablate_transformer, ablate_volume=self.opts.ablate_volume, ablate_res=self.opts.ablate_res)

        if self.opts.use_dp:
            self.models['pose'] = nn.DataParallel(self.models['pose'], device_ids=[i for i in range(self.gpu_count-1)])
            self.models['pose'].cuda()
        else:
            self.models['pose'].to(self.device)

        self.parameters_to_train += list(self.models['pose'].parameters())

        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opts.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opts.scheduler_step_size, 0.7)
        

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
            
            '''
            OpticalFlow(height=self.opts.full_height*self.opts.test_res_opt_flow,
                                        width=self.opts.full_width*self.opts.test_res_opt_flow,
                                        weights_path=self.opts.opt_flow_weights)
            '''
            self.opt_flow.to(self.opt_flow_device)

        if self.opts.use_segmentation_loss:
            self.segmentation = DLV3P()
            self.segmentation.to(self.device)

        if self.opts.load_weights_folder is not None:
            self.load_model()

        if self.opts.use_superglue_eval:
            self.superglue = model.SuperGlueMatcher(self.opts.nms_radius, self.opts.keypoint_threshold, 
                                  self.opts.max_keypoints, self.opts.superglue_weights, self.opts.sinkhorn_iterations,
                                  self.opts.match_threshold)
            self.superglue.to(self.device)
            self.superglue.eval()

        self.loss_fn = torch.nn.L1Loss() if self.opts.loss_fn == "l1" else torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineSimilarity()

        print("experiment number:\n  Training model named:\n  ",
              self.opts.exp_num, self.opts.model_name)
        print("Models and tensorboard events files are saved to:\n  ",
              self.opts.log_dir)
        if self.opts.use_dp:
            print("Training is using:\n ", [i for i in range(self.gpu_count)])
        else:
            print("Training is using:\n ", self.device)

        dataset_dict = {
            "torc": dataset.StereoPoseDataset,
            "carla": dataset.CarlaStereoPoseDataset,
            "carla2": dataset.CarlaStereoPose2Dataset,
            "kitti": dataset.KITTIDataset,
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

        train_dataset = dataset_dict[self.opts.dataset](dir_dict["gt"], dir_dict["data"],
                                                        os.path.join(
                                                            dir_dict["splits"], 'train_dataset.csv'),
                                                        self.opts.height, self.opts.width, True, self.opts.pose_repr,
                                                        self.opts.is_crop, self.opts.resize_orig_img,
                                                        self.opts.perturbation_level)

        val_dataset = dataset_dict[self.opts.dataset](dir_dict["gt"], dir_dict["data"],
                                                      os.path.join(dir_dict["splits"], 'val_dataset.csv'),
                                                      self.opts.height, self.opts.width, False, self.opts.pose_repr,
                                                      self.opts.is_crop, self.opts.resize_orig_img, 
                                                      self.opts.perturbation_level)

        self.num_total_steps = len(
            train_dataset) // self.opts.batch_size * self.opts.num_epochs

        if self.opts.use_subset:
            train_dataset = torch.utils.data.Subset(train_dataset, range(0, 500))
            val_dataset = torch.utils.data.Subset(val_dataset, range(0, 50))
            self.train_loader = DataLoader(
                train_dataset, self.opts.batch_size, False,
                num_workers=self.opts.num_workers, pin_memory=True, drop_last=True)
            self.val_loader = DataLoader(
                val_dataset, self.opts.batch_size, False,
                num_workers=self.opts.num_workers, pin_memory=True, drop_last=True)            
            self.test_loader = DataLoader(
                val_dataset, 1, False,
                num_workers=self.opts.num_workers, pin_memory=True, drop_last=True)            

        else:
            self.train_loader = DataLoader(
                train_dataset, self.opts.batch_size,
                num_workers=self.opts.num_workers, pin_memory=False, drop_last=True, shuffle=True)
            self.val_loader = DataLoader(
                val_dataset, self.opts.batch_size,
                num_workers=self.opts.num_workers, pin_memory=False, drop_last=True, shuffle=True)
            self.test_loader = DataLoader(
                val_dataset, 1,
                num_workers=self.opts.num_workers, pin_memory=False, drop_last=True, shuffle=False)

        self.val_iter = iter(self.val_loader)

        # TensorBoard Logging
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(
                os.path.join(self.log_path, mode))

        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def run_epoch(self):
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            #print(inputs.keys())
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opts.log_frequency == 0 and self.step < 100
            late_phase = self.step % 100 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log_rotation_errors(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1


        if self.opts.use_superglue_eval:
            self.test()
            
        self.model_lr_scheduler.step()


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs, mode="val")
            self.log_rotation_errors(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def test(self):
        """Testing the model on the Validation Set
        """

        self.set_eval()
        print("Testing on SuperGlue")
        
        average_offsets = {}
        for batch_idx, inputs in tqdm(enumerate(self.test_loader)):
            with torch.no_grad():
                outputs, _ = self.process_batch(inputs, mode="test")
                try:
                    self.compute_y_offset(inputs, outputs, average_offsets)
                except IndexError:
                    continue

        writer = self.writers["val"]
        for key, offsets in average_offsets.items():
            writer.add_scalar("{}".format(key), sum(offsets)/len(offsets), self.step)

        self.set_train()

    def compute_y_offset(self, inputs, outputs, offsets):
        """Estimate average keypoint offset in the Y-axis
        """

        of_resize = self.opts.algolux_test_res_opt_flow

        input_matches = self.superglue(inputs['target_aug'], inputs['reference_aug'])
        offsets["superglue_offset/input"] = offsets.get("superglue_offset/input", []) + [torch.abs(input_matches['mkeypoints0'][:, 1] - input_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]

        img1 = F.interpolate(outputs['pred_left_rectf'], scale_factor=(of_resize, of_resize))
        img2 = F.interpolate(outputs['pred_right_rectf'], scale_factor=(of_resize, of_resize))
        pred_matches = self.superglue(img1, img2)
        offsets["superglue_offset/pred"] = offsets.get("superglue_offset/pred", []) + [torch.abs(pred_matches['mkeypoints0'][:, 1] - pred_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]

        gt1 = F.interpolate(outputs['gt_left_rectf'], scale_factor=(of_resize, of_resize))
        gt2 = F.interpolate(outputs['gt_right_rectf'], scale_factor=(of_resize, of_resize))
        gt_matches = self.superglue(gt1, gt2)
        offsets["superglue_offset/gt"] = offsets.get("superglue_offset/gt", []) + [torch.abs(gt_matches['mkeypoints0'][:, 1] - gt_matches['mkeypoints1'][:, 1]).mean().detach().cpu().numpy()]

    def process_batch(self, inputs, mode="train"):
        """Pass a minibatch through the network and generate images and losses
        """
        outputs = {}
        for key, ipt in inputs.items():
            if not key in ["img_id",'dataset']:
                if self.opts.use_dp:
                    inputs[key] = ipt.cuda(non_blocking=True)
                else:
                    inputs[key] = ipt.to(self.device)

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

        self.generate_rectf_imgs(inputs, outputs)
        losses = None
        if mode != "test":
            losses = self.compute_losses(inputs, outputs, mode)

        return outputs, losses

    def generate_rectf_imgs(self, inputs, outputs):
        pred_rotation = outputs["rotation"]
        gt_rotation = inputs["rotation"]

        if self.opts.pose_repr in ['use_6D', 'use_perturb_6D']:
            pred_rmat = gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:])
            gt_rmat = gram_schmidt(gt_rotation[:, :3], gt_rotation[:, 3:])
        elif self.opts.pose_repr in ['use_9D', 'use_perturb', 'use_yaw_perturb', 'use_identity', 'use_yaw']:
            pred_rmat = svd_orthogonalize(pred_rotation.reshape(-1, 3, 3))
            gt_rmat = gt_rotation
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

    def compute_losses(self, inputs, outputs, mode="train"):
        losses = {}
        loss = 0
        gt_rotation = inputs["rotation"]
        pred_rotation = outputs["rotation"]

        of_resize = self.opts.algolux_test_res_opt_flow
        
        if self.opts.use_segmentation_loss and self.epoch >= self.opts.epoch_visual_loss:

            img1 = F.interpolate(outputs['pred_left_rectf'], scale_factor=(of_resize, of_resize))
            img2 = F.interpolate(outputs['pred_right_rectf'], scale_factor=(of_resize, of_resize))
            
            gt1 = F.interpolate(outputs['gt_left_rectf'], scale_factor=(of_resize, of_resize))
            gt2 = F.interpolate(outputs['gt_right_rectf'], scale_factor=(of_resize, of_resize))
            
            pred_mask_left, colorized_mask_left = self.segmentation(img1)
            pred_mask_right, colorized_mask_right = self.segmentation(img2)

            _, gt_mask_left = self.segmentation(gt1)
            _, gt_mask_right = self.segmentation(gt2)

            outputs["pred_left_seg"] = colorized_mask_left
            outputs["pred_right_seg"] = colorized_mask_right

            outputs["gt_left_seg"] = gt_mask_left
            outputs["gt_right_seg"] = gt_mask_right

            b, h, w = pred_mask_left.shape
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            grid_y = grid_y.unsqueeze(0).repeat(b, 1, 1).float().cuda()
            grid_y = grid_y / h

            losses["segmentation"] = 0
            for classes in [0, 1, 10, 13, 8]:
                class_ids_left = torch.where(pred_mask_left==classes, 1.0, 0.0)
                class_ids_right = torch.where(pred_mask_right==classes, 1.0, 0.0)
            
                y_coord_left = (grid_y*class_ids_left).mean(dim=2).mean(dim=1)
                y_coord_right = (grid_y*class_ids_right).mean(dim=2).mean(dim=1)

                losses["segmentation"] += torch.abs(y_coord_left-y_coord_right).mean()

            loss += 100*losses["segmentation"]

        if self.opts.pose_repr in ['use_6D', 'use_perturb_6D']:
            pred_rmat = gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:])
            gt_rmat = gram_schmidt(gt_rotation[:, :3], gt_rotation[:, 3:])
        elif self.opts.pose_repr in ['use_9D', 'use_perturb', 'use_identity', 'use_yaw_perturb', 'use_yaw']:
            pred_rmat = svd_orthogonalize(pred_rotation.reshape(-1, 3, 3))
            gt_rmat = gt_rotation
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

        if self.opts.pose_repr == "use_rectify":
            losses['rotation'] = self.loss_fn(gt_r1.double(), pred_r1.double()) + self.loss_fn(gt_r2.double(), pred_r2.double())
        else:
            losses['rotation'] = self.loss_fn(matrix_to_euler_angles(gt_rmat.double(),'XYZ'), 
                                              matrix_to_euler_angles(pred_rmat.double(),'XYZ'))
            
            if self.opts.use_seq_hsm and self.opts.all_rotation_loss:
                for predictions in outputs["predicted_rotations"][:-1]:
                    predict_svd = svd_orthogonalize(predictions.reshape(-1, 3, 3))
                    losses['rotation'] += self.loss_fn(matrix_to_euler_angles(gt_rmat.double(),'XYZ'), 
                                                       matrix_to_euler_angles(predict_svd.double(),'XYZ'))

        loss += self.opts.rotation * losses['rotation']

        if self.opts.use_I_loss:
            nsample = pred_rmat.shape[0]
            I = torch.eye(3).unsqueeze(0).to(pred_rmat.device).double()
            I = I.repeat(nsample, 1, 1)
            I_euler = matrix_to_euler_angles(I.double(),'XYZ')
            I_perturb = I_euler + torch.normal(0, 1.0, size=I_euler.shape).to(pred_rmat.device)

            losses['identity_loss'] = self.loss_fn(I_perturb, 
                                                   matrix_to_euler_angles(pred_rmat.double(),'XYZ'))
            loss += losses['identity_loss']

        if self.opts.use_cost_vol_loss and self.epoch >= self.opts.epoch_visual_loss:

            img1 = F.interpolate(outputs['pred_left_rectf'], size=(self.opts.height, self.opts.width))
            img2 = F.interpolate(outputs['pred_right_rectf'], size=(self.opts.height, self.opts.width))
            _, cost_vol3_dict = self.models["pose"](img1, img2)
            losses['cost_vol_horizontal'] = torch.min(cost_vol3_dict["horizontal_disp"], dim=2)[0].mean()
            losses['cost_vol_vertical'] = torch.argmin(cost_vol3_dict["vertical_disp"], dim=2).float().mean()
            #print(outputs['cost_vol3_dict']["horizontal_disp"].shape, losses['cost_vol_vertical'], losses['cost_vol_horizontal'])
            loss += losses['cost_vol_vertical']

        if self.opts.use_new_costvol_loss and self.epoch >= self.opts.epoch_visual_loss:
            translation = inputs['translation'].unsqueeze(-1)

            intrinsics_left, intrinsics_right = inputs["intrns_left"], inputs['intrns_right']

            dist1 = inputs['dist_coeff1']
            dist2 = inputs['dist_coeff2']

            b, c, h, w = outputs['cost_vol3_dict']['conv10'].shape
            outputs['conv_10_rectf'], outputs['conv_11_rectf'], _ = \
                calibrate_stereo_pair_torch(
                    outputs['cost_vol3_dict']['conv10'], outputs['cost_vol3_dict']['conv11'], intrinsics_left, intrinsics_right, dist1, dist2, pred_rmat.double(), translation.double(),
                    h, w)

            feats = self.models["pose"].get_cost_vol(feat_left=outputs['conv_10_rectf'], 
                                                     feat_right=outputs['conv_11_rectf'], 
                                                     hor_conv3d=self.models["pose"].conv_3d_3, 
                                                     vert_conv3d=self.models["pose"].conv_3d_3_vert,
                                                     vertical=True,
                                                     )
            
            losses['cost_vol_horizontal'] = torch.min(feats[0], dim=2)[0].mean()
            losses['cost_vol_vertical'] = torch.argmin(feats[1], dim=2).float().mean()

            loss += losses['cost_vol_vertical']

        if self.opts.use_dispwarp_feat_loss:
            translation = inputs['translation'].unsqueeze(-1)

            intrinsics_left, intrinsics_right = inputs["intrns_left"], inputs['intrns_right']

            dist1 = inputs['dist_coeff1']
            dist2 = inputs['dist_coeff2']

            b, c, h, w = outputs['cost_vol3_dict']['conv10'].shape
            outputs['conv_10_rectf'], outputs['conv_11_rectf'], _ = \
                calibrate_stereo_pair_torch(
                    outputs['cost_vol3_dict']['conv10'], outputs['cost_vol3_dict']['conv11'], intrinsics_left, intrinsics_right, dist1, dist2, pred_rmat.double(), translation.double(),
                    h, w)

            feats = self.models["pose"].get_cost_vol(feat_left=outputs['conv_10_rectf'], 
                                                     feat_right=outputs['conv_11_rectf'], 
                                                     hor_conv3d=self.models["pose"].conv3d_3, 
                                                     vert_conv3d=self.models["pose"].conv3d_3_vert,
                                                     vertical=True,
                                                     pool=False
                                                     )
            
            disp_lr = self.models["pose"].wta_disparity(feats[0].squeeze())
            outputs['conv_10_warp'] = self.models["pose"].interpolate_left_to_right(left_image=outputs['conv_10_rectf'], 
                                                                                    right_image=outputs['conv_11_rectf'], 
                                                                                    disparity=disp_lr)

            losses["dispwarp_feat"] = torch.abs(outputs['conv_10_warp'] - outputs['conv_11_rectf']).mean()
            loss += 100*losses['dispwarp_feat']

        if self.opts.use_dispwarp_img_loss and self.epoch >= self.opts.epoch_visual_loss:

            img1 = F.interpolate(outputs['pred_left_rectf'], scale_factor=(of_resize, of_resize))
            img2 = F.interpolate(outputs['pred_right_rectf'], scale_factor=(of_resize, of_resize))

            img_vol = self.models["pose"].feature_vol(img1, 
                                                      img2,
                                                      maxdisp=96,
                                                      )
            
            #print(img_vol[0].mean(axis=2).shape, img_vol[1].mean(axis=2)[:10])

            disp_lr = self.models["pose"].wta_disparity(img_vol[0][:, 1:2, ...].squeeze())

            img1_warp = self.models["pose"].interpolate_left_to_right(left_image=img1, 
                                                                      right_image=img2, 
                                                                      disparity=disp_lr)

            losses["dispwarp_img"] = torch.abs(img1_warp - img2).mean()
            loss += 100*losses['dispwarp_img']

        if self.opts.use_dispfeatvol_loss:
            translation = inputs['translation'].unsqueeze(-1)

            intrinsics_left, intrinsics_right = inputs["intrns_left"], inputs['intrns_right']

            dist1 = inputs['dist_coeff1']
            dist2 = inputs['dist_coeff2']

            b, c, h, w = outputs['cost_vol3_dict']['conv10'].shape
            width_scale, height_scale = w/inputs['full_width'][0], h/inputs['full_height'][0]
            scale_mat = torch.tensor([[[width_scale, 0, width_scale], [0, height_scale, height_scale], [0, 0, 1]]])
            scale_mat = scale_mat.to(intrinsics_left.device)

            outputs['conv_10_rectf'], outputs['conv_11_rectf'], _ = calibrate_stereo_pair_torch(outputs['cost_vol3_dict']['conv10'], 
                                                                                                outputs['cost_vol3_dict']['conv11'], 
                                                                                                scale_mat*intrinsics_left, 
                                                                                                scale_mat*intrinsics_right, 
                                                                                                dist1, dist2, 
                                                                                                pred_rmat.double(), translation.double(), 
                                                                                                h, w)

            featvol_rectf = self.models["pose"].feature_vol(outputs['conv_10_rectf'], 
                                                            outputs['conv_11_rectf'],
                                                            maxdisp=3)
            
            _, costvol_rectf = self.models["pose"].decoder6(featvol_rectf)
            
            outputs["disp_lr"] = self.models["pose"].wta_disparity(F.softmax(costvol_rectf, 1))
            
            #print(outputs["disp_lr"].min(), outputs["disp_lr"].max())
            #print(outputs["conv_10_rectf"].min(), outputs["conv_10_rectf"].max())
            #print(outputs["conv_11_rectf"].min(), outputs["conv_11_rectf"].max())

            outputs['conv_10_warp'] = self.models["pose"].interpolate_left_to_right(outputs['conv_10_rectf'], 
                                                                                    outputs['conv_11_rectf'], 
                                                                                    outputs["disp_lr"])
            
            #print(outputs["conv_10_warp"].min(), outputs["conv_10_warp"].max())
            
            outputs['featvol_vert'] = self.models["pose"].feature_vol(outputs['conv_10_warp'].permute(0, 1, 3, 2), 
                                                                      outputs['conv_11_rectf'].permute(0, 1, 3, 2), 
                                                                      maxdisp=1)

            losses["dispfeatvol_loss"] = outputs['featvol_vert'].mean() + torch.abs(outputs['conv_10_warp'] - outputs['conv_11_rectf']).mean()
            loss += (losses["dispfeatvol_loss"]*10)

        if self.opts.y_axis_euler_loss and self.opts.pose_repr != "use_rectify":
            losses['y_axis'] = self.loss_fn(matrix_to_euler_angles(gt_rmat.double(),'XYZ')[..., 1:2], 
                                            matrix_to_euler_angles(pred_rmat.double(),'XYZ')[..., 1:2])
            loss += self.opts.rotation * losses['y_axis'] * 2

        if self.opts.use_epipole_loss:
            losses['epipole_dist'], losses['epipole_line_angle'] = self.epipolar_loss(pred_rmat, gt_rmat, inputs)
            losses['epipole_loss'] = 1e-5*losses['epipole_dist'] + 100*losses['epipole_line_angle']
            loss += losses['epipole_loss']

        if self.opts.use_trace_similarity_loss:
            losses['trace_similarity'] = self.trace_similarity_loss(pred_rmat, gt_rmat)
            loss += losses['trace_similarity']

        if self.opts.use_dir_loss:
            if self.opts.pose_repr in ['use_perturb', 'use_9D', 'use_yaw_perturb', 'use_identity', 'use_yaw']:
                losses['direction_loss'] = ((1-self.cosine_loss(gt_rotation[:, 0, :], pred_rotation[:, 0, :])) +
                                            (1-self.cosine_loss(gt_rotation[:, 1, :], pred_rotation[:, 1, :])) +
                                            (1-self.cosine_loss(gt_rotation[:, 2, :], pred_rotation[:, 2, :]))).mean()
            elif self.opts.pose_repr in ['use_perturb_quat', 'use_quat']:
                losses['direction_loss'] = 1-self.cosine_loss(gt_rotation, pred_rotation).mean()
            else:
                losses['direction_loss'] = (self.direction_loss(gt_rotation[:, :3], pred_rotation[:, :3]) +
                                        self.direction_loss(gt_rotation[:, 3:], pred_rotation[:, 3:]))

            loss += self.opts.rotation * losses['direction_loss']
            
        if  self.opts.use_opt_flow_loss and self.epoch >= self.opts.epoch_visual_loss:
            
            of_height, of_width = self.opts.height, self.opts.width

            if self.opts.use_opt_flow_random_crop:
                #If error line below is broken fix issue if using random crops Width and full_width are the same
                x, y = torch.randint(0, inputs['full_width'][0]-self.opts.width, (1, )), torch.randint(0, inputs['full_height'][0]-self.opts.height, (1, ))

                img1 = outputs['pred_left_rectf'][:, :, y:y+self.opts.height, x:x+self.opts.width]
                img2 = outputs['pred_right_rectf'][:, :, y:y+self.opts.height, x:x+self.opts.width]
            else:
                # downsample the image for calculation of optical flow
                img1 = F.interpolate(outputs['pred_left_rectf'],scale_factor=(of_resize, of_resize))
                img2 = F.interpolate(outputs['pred_right_rectf'],scale_factor=(of_resize, of_resize))
                of_height, of_width = img1.shape[2], img1.shape[3]
            # create joint mask of ROI for left and right images and downsample it 
            joint_mask = inputs['mask_left'] & inputs['mask_right']
            if self.opts.use_opt_flow_random_crop:
                joint_mask = joint_mask.double().unsqueeze(1)[:, :, y:y+self.opts.height, x:x+self.opts.width]
            else:
                joint_mask = F.interpolate(joint_mask.double().unsqueeze(1),scale_factor=(of_resize, of_resize))

            # calculate optical flow 
            flow_x, flow_y = self.opt_flow(img1.to(self.opt_flow_device), img2.to(self.opt_flow_device))
            if self.opts.use_sky_mask:
                flow_x, flow_y = flow_x[:, of_height//3:, :], flow_y[:, of_height//3:, :]

            flow_x = torch.nan_to_num(flow_x, nan = 0.0)

            assert not (torch.any(flow_x.isnan()) or torch.any(flow_y.isnan())), "NaN values detected in optical flow calculation"

            if self.opts.use_radial_flow_mask:
                radial_mask_full = inputs["radial_mask_full"].to(self.opt_flow_device)

                if self.opts.use_opt_flow_random_crop:
                    radial_mask = radial_mask_full[:, :, y:y+self.opts.height, x:x+self.opts.width]
                else:
                    radial_mask = F.interpolate(radial_mask_full, scale_factor=(of_resize, of_resize))
                    if self.opts.use_sky_mask:
                        radial_mask = radial_mask[:, :, of_height//3:, :]

                inputs["radial_mask"] = radial_mask
                losses['opt_flow'] = (radial_mask*torch.abs(flow_y)).mean()
            else:
                losses['opt_flow'] = torch.abs(flow_y).mean()

            # losses['opt_flow'] = (torch.abs(flow_y)*joint_mask.type_as(flow_y)).mean()
                
            losses['negative_opt_flow'] = torch.count_nonzero(flow_x > 0) / (inputs['full_height'][0]*inputs['full_width'][0]).to(flow_x.device)

            assert not losses['opt_flow'].isnan(), "optical flow loss is nan"
            loss += (losses['opt_flow']/10).to(self.device)

            if not losses['negative_opt_flow'].isnan():
                loss += (losses['negative_opt_flow'].to(self.device)*10)

            # store optical flow for logging
            outputs['flow_x'] = flow_x.detach().cpu().numpy()
            outputs['flow_y'] = flow_y.detach().cpu().numpy()
            outputs['joint_mask'] = joint_mask.cpu().numpy()

        if self.opts.constrain_roi:
            black_pixels_left = outputs['pred_left_rectf'].mean(dim=1) == 0
            black_pixels_right = outputs['pred_right_rectf'].mean(dim=1) == 0
            black_pixels = torch.count_nonzero(black_pixels_left) + torch.count_nonzero(black_pixels_right)

            losses['black_pixel_count'] = black_pixels / (inputs['full_height'][0]*inputs['full_width'][0]).to(black_pixels.device)

            loss += (losses['black_pixel_count'])

        if self.opts.use_visual_loss and self.epoch >= self.opts.epoch_visual_loss:
           pred_img1 = outputs['pred_left_rectf']
           pred_img2 = outputs['pred_right_rectf']

           gt_img1 = outputs['gt_left_rectf']
           gt_img2 = outputs['gt_right_rectf']

           losses['visual_loss'] =  (torch.abs(pred_img1 - gt_img1)+torch.abs(pred_img2 - gt_img2)).mean()
           
           outputs['visual_diff_left'] = torch.abs(pred_img1 - gt_img1).detach().cpu().numpy()
           outputs['visual_diff_right'] = torch.abs(pred_img2 - gt_img2).detach().cpu().numpy()
           
           loss +=  losses['visual_loss']

        if self.opts.use_visual_loss_self and self.epoch >= self.opts.epoch_visual_loss:
           pred_img1 = outputs['pred_left_rectf']
           pred_img2 = outputs['pred_right_rectf']

           joint_mask = inputs['mask_left'] & inputs['mask_right']

           pred_mask_img1 = joint_mask * pred_img1
           pred_mask_img2 = joint_mask * pred_img2

           gt_img1 = outputs['gt_left_rectf']
           gt_img2 = outputs['gt_right_rectf']

           gt_mask_img1 = joint_mask * gt_img1
           gt_mask_img2 = joint_mask * gt_img2

           losses['visual_loss_self'] =  torch.abs(pred_mask_img1 - pred_mask_img2).mean()
           
           outputs['visual_diff_pred'] = torch.abs(pred_mask_img1 - pred_mask_img2).detach().cpu().numpy()
           outputs['visual_diff_gt'] = torch.abs(gt_mask_img1 - gt_mask_img2).detach().cpu().numpy()
           
           loss +=  losses['visual_loss_self']

        if self.opts.model_name == 'posecnn':
            gt_translation = inputs['translation']
            pred_translation = outputs["translation"].reshape(
                gt_translation.shape)
            losses['translation'] = self.loss_fn(
                pred_translation, gt_translation)
            loss += losses['translation']

        losses["loss"] = loss

        return losses

    def skewmat(self, x_vec):
        '''
        torch.matrix_exp(a)
        Eigen::Matrix3f mat = Eigen::Matrix3f::Zero();
        mat(0, 1) = -v[2]; mat(0, 2) = +v[1];
        mat(1, 0) = +v[2]; mat(1, 2) = -v[0];
        mat(2, 0) = -v[1]; mat(2, 1) = +v[0];
        return mat;
        input : (*, 3)
        output : (*, 3, 3)
        '''

        W_row0 = torch.tensor([0,0,0,  0,0,1,  0,-1,0]).view(3,3).to(x_vec.device).float()
        W_row1  = torch.tensor([0,0,-1,  0,0,0,  1,0,0]).view(3,3).to(x_vec.device).float()
        W_row2  = torch.tensor([0,1,0,  -1,0,0,  0,0,0]).view(3,3).to(x_vec.device).float()

        x_skewmat = torch.stack([torch.matmul(x_vec, W_row0.t()) , torch.matmul(x_vec, W_row1.t()), torch.matmul(x_vec, W_row2.t())] , dim = -1)

        return x_skewmat

    def epipolar_loss(self, r_pred, r_true, inputs):
        b = inputs['target'].shape[0]

        zeros = torch.zeros((1, 3, 1))
        identity = torch.eye(3).reshape(1, 3, 3)
        zeros = zeros.repeat(b, 1, 1)
        identity = identity.repeat(b, 1, 1)

        extr_left = torch.cat((identity, zeros), axis=-1)
        P_left = torch.bmm(inputs['intrns_left'].float(), extr_left.to(self.device))

        inverse_trans = torch.bmm(-r_pred.permute(0, 2, 1).float(), inputs['translation'].unsqueeze(-1))
        ones = torch.ones((1, 1, 1))
        ones = ones.repeat(b, 1, 1)
        inverse_trans_homo = torch.cat((inverse_trans, ones.to(self.device)), axis=-2)

        extr_right = torch.cat((r_pred, inputs["translation"].unsqueeze(-1)), axis=-1)
        P_right = torch.bmm(inputs['intrns_right'].float(), extr_right.to(self.device).float())
        trans_right = torch.tensor((0, 0, 0, 1)).to(self.device).unsqueeze(0)
        trans_right = trans_right.repeat(b, 1, 1).float()

        epipole_r2l = torch.bmm(P_left, inverse_trans_homo)
        epipole_r2l_euc = epipole_r2l[:, :2, :] / epipole_r2l[:, 2:, :]
        epipole_l2r = torch.bmm(P_right, trans_right.permute(0, 2, 1))
        epipole_l2r_euc = epipole_l2r[:, :2, :] / epipole_l2r[:, 2:, :]

        epipole_dist = -1*torch.norm(epipole_l2r_euc.squeeze()-epipole_r2l_euc.squeeze())

        epipolar_line_angle = torch.atan((epipole_l2r_euc[:, 1, :]-epipole_r2l_euc[:, 1, :]) / (epipole_l2r_euc[:, 0, :]-epipole_r2l_euc[:, 0, :])).mean()
        
        #epipolar_line = torch.cat((epipole_r2l_euc[:, 0:1, :]-epipole_l2r_euc[:, 0:1, :], epipole_r2l_euc[:, 1:, :]-epipole_l2r_euc[:, 1:, :]), axis=-2)
        #x_line = torch.tensor((1, 0)).to(self.device).unsqueeze(0)
        #x_line = x_line.repeat(b, 1, 1).permute(0, 2, 1)

        #cosine_loss = 1 - self.cosine_loss(epipolar_line, x_line)
        #print(epipole_dist, epipolar_line_angle, cosine_loss)

        return epipole_dist, epipolar_line_angle

    def trace_similarity_loss(self, r_A, r_B):
        r_A_t = r_A.permute(0, 2, 1)
        r_AB = torch.bmm(r_A_t.float(), r_B.float())
        trace_r_AB = r_AB.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        if trace_r_AB > 3:
            trace_r_AB = -1*torch.ones(trace_r_AB.shape)
        theta = torch.nan_to_num(torch.acos((trace_r_AB - 1) / 2), nan = 3.14).mean()

        return theta

    def direction_loss(self, v_pred, v_true):
        """The direction loss measures the negative cosine similarity between vectors.
        Args:
            v_pred: [BATCH, 3] predicted unit vectors.
            v_true: [BATCH, 3] ground truth unit vectors.
        Returns:
            A float scalar.
        """
        return -torch.mean(torch.sum(v_pred * v_true, dim=-1))

    def compute_rotation_errors(self, pred_rotation, gt_axisangles):

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

        delta = np.absolute(gt_axisangles - pred_axisangles)

        return delta


    def log_rotation_errors(self, inputs, outputs, losses):
        gt_axisangles = inputs['axisangles'].cpu().numpy()
        pred_rotation = outputs["rotation"].detach().cpu()

        if self.opts.pose_repr == "use_rectify":        
            pred_r1, pred_r2 = outputs["R1"], outputs["R2"]
            gt_r1, gt_r2 = inputs["R1"], inputs["R2"]
            P1, P2 = inputs["P1"], inputs["P2"]
            pred1_axisangles = Rotation.from_matrix(gram_schmidt(pred_r1[:, :3], pred_r1[:, 3:]).detach().cpu().numpy()).as_rotvec(degrees=True).astype(np.float32)
            pred2_axisangles = Rotation.from_matrix(gram_schmidt(pred_r2[:, :3], pred_r2[:, 3:]).detach().cpu().numpy()).as_rotvec(degrees=True).astype(np.float32)
            gt1_axisangles = Rotation.from_matrix(gt_r1.cpu().numpy()).as_rotvec(degrees=True).astype(np.float32)
            gt2_axisangles = Rotation.from_matrix(gt_r2.cpu().numpy()).as_rotvec(degrees=True).astype(np.float32)

            delta1 = np.absolute(gt1_axisangles - pred1_axisangles)
            delta2 = np.absolute(gt2_axisangles - pred2_axisangles)

            losses['axisangle(degrees)/delta1_x'] = delta1[:, 0].mean()
            losses['axisangle(degrees)/delta1_y'] = delta1[:, 1].mean()
            losses['axisangle(degrees)/delta1_z'] = delta1[:, 2].mean()
            losses['axisangle(degrees)/delta2_x'] = delta2[:, 0].mean()
            losses['axisangle(degrees)/delta2_y'] = delta2[:, 1].mean()
            losses['axisangle(degrees)/delta2_z'] = delta2[:, 2].mean()

        else:
            delta = self.compute_rotation_errors(pred_rotation, gt_axisangles)

            losses['axisangle(degrees)/delta_x'] = delta[:, 0].mean()
            losses['axisangle(degrees)/delta_y'] = delta[:, 1].mean()
            losses['axisangle(degrees)/delta_z'] = delta[:, 2].mean()

        if self.opts.use_seq_hsm:
            for i, pred_rot in enumerate(outputs["predicted_rotations"]):
                delta = self.compute_rotation_errors(pred_rot.detach().cpu(), gt_axisangles)
                losses["rot_{}_axisangle(degrees)/delta_x".format(i)] = delta[:, 0].mean()
                losses["rot_{}_axisangle(degrees)/delta_y".format(i)] = delta[:, 1].mean()
                losses["rot_{}_axisangle(degrees)/delta_z".format(i)] = delta[:, 2].mean()


    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        if "siamese" in self.opts.model_name:
            for i, (layer, param) in enumerate(self.models["pose"].named_parameters()):
                if param.requires_grad:
                    writer.add_histogram("{}_grad".format(layer), param.grad, self.step) 
                if i > 4:
                    break

        # Generate rectified images from both ground truth and predicted poses
        pred_rotation = outputs["rotation"].detach().cpu()
        gt_rotation = inputs["rotation"].detach().cpu()
        if self.opts.pose_repr in ['use_perturb', 'use_perturb_quat', 'use_perturb_6D', 'use_identity','use_yaw_perturb']:
            init_rotation = inputs["rotation_perturb"].detach().cpu()

        if self.opts.pose_repr in ['use_6D', 'use_perturb_6D']:
            pred_rmat = Rotation.from_matrix(gram_schmidt(
                pred_rotation[:, :3], pred_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            gt_rmat = Rotation.from_matrix(gram_schmidt(
                gt_rotation[:, :3], gt_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)
            if self.opts.pose_repr == 'use_perturb_6D':
                init_rmat = Rotation.from_matrix(gram_schmidt(
                    init_rotation[:, :3], init_rotation[:, 3:]).numpy()).as_matrix().astype(np.float32)

        elif self.opts.pose_repr in ['use_9D', 'use_perturb', 'use_identity', 'use_yaw_perturb', 'use_yaw']:
            pred_rotation = pred_rotation.reshape(-1, 3, 3)
            pred_rmat = svd_orthogonalize(pred_rotation).numpy().astype(np.float32)
            gt_rmat = gt_rotation.numpy().astype(np.float32)
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

        translation = inputs['translation'].detach().cpu().numpy()

        left_imgs = inputs['full_left'].permute(0, 2, 3, 1).cpu().numpy()
        right_imgs = inputs['full_right'].permute(0, 2, 3, 1).cpu().numpy()


        for j in range(min(4, self.opts.batch_size)):
            unrect_left_img = (left_imgs[j, ...] *255.0).astype(np.uint8)
            unrect_right_img = (right_imgs[j, ...] *255.0).astype(np.uint8)

            height,width,_ = unrect_left_img.shape

            if self.opts.pose_repr != "use_rectify":
                R_pred = pred_rmat[j,...]
                R_gt = gt_rmat[j,...]
            
            t = translation[j,...]

            K_left, K_right = inputs["intrns_left"][j].cpu().numpy(), inputs['intrns_right'][j].cpu().numpy()

            dist1 = inputs['dist_coeff1'][j].cpu().numpy()
            dist2 = inputs['dist_coeff2'][j].cpu().numpy()

            if self.opts.pose_repr == "use_rectify":
                pred_img_left_rect, pred_img_right_rect = undistortrectify_cv2(
                    unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), pred_r1[j, ...].astype(np.float64), pred_r2[j, ...].astype(np.float64), P1[j, ...].astype(np.float64), P2[j, ...].astype(np.float64), height, width)
                gt_img_left_rect, gt_img_right_rect = undistortrectify_cv2(
                    unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), gt_r1[j, ...].astype(np.float64), gt_r2[j, ...].astype(np.float64), P1[j, ...].astype(np.float64), P2[j, ...].astype(np.float64), height, width)

            else:
                pred_img_left_rect, pred_img_right_rect, _ = calibrate_stereo_pair(
                    unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_pred.astype(np.float64), t.astype(np.float64), height, width)

                gt_img_left_rect, gt_img_right_rect, _ = calibrate_stereo_pair(
                    unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_gt.astype(np.float64), t.astype(np.float64), height, width)

            pred_ep_image_pair = draw_epipolar_lines(pred_img_left_rect, pred_img_right_rect, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
            gt_ep_image_pair = draw_epipolar_lines(gt_img_left_rect, gt_img_right_rect, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
            input_ep_image_pair = draw_epipolar_lines(unrect_left_img.copy(), unrect_right_img.copy(), num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])

            pred_image_overlay = cv2.addWeighted(pred_img_left_rect, 0.5, 
                                                 pred_img_right_rect, 0.5, 0)

            gt_image_overlay = cv2.addWeighted(gt_img_left_rect, 0.5, 
                                               gt_img_right_rect, 0.5, 0)

            input_image_overlay = cv2.addWeighted(unrect_left_img, 0.5, 
                                                  unrect_right_img, 0.5, 0)



            ### Visualize initialization
            if self.opts.pose_repr in ['use_perturb', 'use_perturb_quat', 'use_perturb_6D', 'use_identity', 'use_yaw_perturb']:
                R_init = init_rmat[j,...]

                init_img_left_rect, init_img_right_rect, _ = calibrate_stereo_pair(
                    unrect_left_img, unrect_right_img, K_left.astype(np.float64), K_right.astype(np.float64), dist1.astype(np.float64), dist2.astype(np.float64), R_init.astype(np.float64), t.astype(np.float64), height, width)

                init_ep_image_pair = draw_epipolar_lines(init_img_left_rect, init_img_right_rect,num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])

                init_image_overlay = cv2.addWeighted(init_img_left_rect, 0.5, 
                                                      init_img_right_rect, 0.5, 0)

                writer.add_image("init_epipolar_images/{}".format(j), init_ep_image_pair, self.step, dataformats='HWC')
                writer.add_image("init_overlay/{}".format(j), init_image_overlay, self.step, dataformats='HWC')

            if self.opts.dataset in ['flyingthings', 'dstereo', 'sintel', 'hd1k']:
                left_ogs = inputs['og_left'].permute(0, 2, 3, 1).cpu().numpy()
                right_ogs = inputs['og_right'].permute(0, 2, 3, 1).cpu().numpy()

                left_og = (left_ogs[j, ...] *255.0).astype(np.uint8)
                right_og = (right_ogs[j, ...] *255.0).astype(np.uint8)

                og_image_overlay = cv2.addWeighted(left_og, 0.5, 
                                                    right_og, 0.5, 0)
                
                writer.add_image("init_overlay/{}".format(j), og_image_overlay, self.step, dataformats='HWC')


            scale_percent = 50 # percent of original size
            width = int(pred_ep_image_pair.shape[1] * scale_percent / 100)
            height = int(pred_ep_image_pair.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            pred_ep_image_pair = cv2.resize(pred_ep_image_pair, dim, interpolation = cv2.INTER_AREA)
            gt_ep_image_pair = cv2.resize(gt_ep_image_pair, dim, interpolation = cv2.INTER_AREA)

            # cv2.imwrite('prediction.png',pred_ep_image_pair)
            # cv2.imwrite('gt.png',gt_ep_image_pair)
            writer.add_image("input_epipolar_images/{}".format(j), input_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("pred_epipolar_images/{}".format(j), pred_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("gt_epipolar_images/{}".format(j), gt_ep_image_pair, self.step, dataformats='HWC')
            writer.add_image("input_overlay/{}".format(j), input_image_overlay, self.step, dataformats='HWC')
            writer.add_image("pred_overlay/{}".format(j), pred_image_overlay, self.step, dataformats='HWC')
            writer.add_image("gt_overlay/{}".format(j), gt_image_overlay, self.step, dataformats='HWC')

            # optical flow visualization
            if self.opts.use_opt_flow_loss and self.epoch >= self.opts.epoch_visual_loss:
                flow_x = np.clip(outputs['flow_x'][j],0.0,255.0)
                flow_y = np.clip(outputs['flow_y'][j],0.0,255.0)
                f = np.concatenate((flow_x[...,None],flow_y[...,None]),axis=-1)
                opt_flow_img = flow_to_image(f)
                writer.add_image("optical_flow/{}".format(j), opt_flow_img, self.step, dataformats='HWC')

                # show ROI mask
                joint_mask = outputs['joint_mask'][j]
                writer.add_image("ROI mask/{}".format(j), joint_mask, self.step, dataformats='CHW')

                # show Radial Mask
                if self.opts.use_radial_flow_mask:            
                    radial_mask = inputs["radial_mask"].to(self.opt_flow_device)[0]
                    writer.add_image("Radial mask", radial_mask / radial_mask.max(), self.step, dataformats='CHW')
            
            if self.opts.use_visual_loss and self.epoch >= self.opts.epoch_visual_loss:
                left_diff = outputs['visual_diff_left'][j]
                right_diff = outputs['visual_diff_right'][j]

                writer.add_image("visual_diff_left/{}".format(j), left_diff, self.step, dataformats='CHW')
                writer.add_image("visual_diff_right/{}".format(j), right_diff, self.step, dataformats='CHW')

            if self.opts.use_visual_loss_self and self.epoch >= self.opts.epoch_visual_loss:
                pred_diff = outputs['visual_diff_pred'][j]
                gt_diff = outputs['visual_diff_gt'][j]

                writer.add_image("visual_diff_pred/{}".format(j), pred_diff, self.step, dataformats='CHW')
                writer.add_image("visual_diff_gt/{}".format(j), gt_diff, self.step, dataformats='CHW')

            '''
            if self.opts.use_dispwarp_loss:
                left_warp = outputs['gt_left_warp'][j].detach().cpu().numpy()
                right_img = outputs['gt_right_rectf'][j].detach().cpu().numpy()
                left_img = outputs['gt_left_rectf'][j].detach().cpu().numpy()

                warp_overlay = cv2.addWeighted(left_warp, 0.5, right_img, 0.5, 0)
                rectf_overlay = cv2.addWeighted(left_img, 0.5, right_img, 0.5, 0)
                rectf_image_pair = np.concatenate((left_img, right_img), axis=1) #draw_epipolar_lines(left_img, right_img, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
                warp_image_pair = np.concatenate((left_warp, right_img),axis=1) #draw_epipolar_lines(left_warp, right_img, num_lines=50, line_color=(0,0,0), text=inputs['img_id'][j])
                
                writer.add_image("warp_overlay/{}".format(j), warp_overlay, self.step, dataformats='CHW')
                writer.add_image("rectf_overlay/{}".format(j), rectf_overlay, self.step, dataformats='CHW')
                writer.add_image("warp_pair/{}".format(j), warp_image_pair, self.step, dataformats='CHW')
                writer.add_image("rectf_pair/{}".format(j), rectf_image_pair, self.step, dataformats='CHW')
            '''

            if self.opts.use_dispfeatvol_loss:
                disp_lr = outputs["disp_lr"][j].detach().cpu().numpy()
                writer.add_image("disprity_feat6/{}".format(j), disp_lr, self.step, dataformats='CHW')

            if self.opts.use_segmentation_loss and self.epoch >= self.opts.epoch_visual_loss:
                pred_left_seg = outputs['pred_left_seg']
                pred_right_seg = outputs['pred_right_seg']

                gt_left_seg = outputs['gt_left_seg']
                gt_right_seg = outputs['gt_right_seg']

                pred_seg_image_overlay = cv2.addWeighted(pred_left_seg, 0.5, pred_right_seg, 0.5, 0)
                gt_seg_image_overlay = cv2.addWeighted(gt_left_seg, 0.5, gt_right_seg, 0.5, 0)

                writer.add_image("pred_segmentation_overlay/{}".format(0), pred_seg_image_overlay, self.step, dataformats='HWC')
                writer.add_image("gt_segmentation_overlay/{}".format(0), gt_seg_image_overlay, self.step, dataformats='HWC')