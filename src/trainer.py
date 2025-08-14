import os
import time
from datetime import datetime

import cv2
import dataset
import model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from base_trainer import BaseExp
from pytorch3d.transforms import (
    matrix_to_euler_angles,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utility.data_utils import (
    calibrate_stereo_pair,
    calibrate_stereo_pair_torch,
    draw_epipolar_lines,
    gram_schmidt,
    svd_orthogonalize,
    undistortrectify,
    undistortrectify_cv2,
)
from utility.log_utils import flow_to_image
from utility.opticalflow import TorchvisionRAFT

torch.autograd.set_detect_anomaly(True)


class Experiment(BaseExp):
    def __init__(self, options) -> None:
        super().__init__()
        self.opts = options
        self.log_path = os.path.join(
            self.opts.log_dir, self.opts.exp_name, f"exp-{self.opts.exp_num}_{self.opts.exp_metainfo}"
        )

        if not self.opts.use_dp:
            self.device = torch.device("cpu" if self.opts.no_cuda else "cuda:0")

        self.gpu_count = torch.cuda.device_count()

        if self.opts.use_opt_flow_loss:
            if self.gpu_count > 1 and self.opts.multi_gpu_opt_flow:
                self.opt_flow_device = torch.device(self.gpu_count - 1)
            else:
                self.opt_flow_device = self.device

        self.models = {}
        self.parameters_to_train = []

        self.num_parameters = {
            # 3 parameters for rotation and 3 parameters for transalation
            "use_3D": [3, 3],
            # 4 parameters for rotation and 3 parameters for transalation
            "use_quat": [4, 3],
            # 6 parameters for rotation and 3 parameters for transalation
            "use_6D": [6, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            "use_9D": [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            "use_perturb": [9, 3],
            # 6 parameters for rotation and 3 parameters for transalation
            "use_perturb_6D": [6, 3],
            # 4 parameters for rotation and 3 parameters for transalation
            "use_perturb_quat": [4, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            "use_identity": [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            "use_yaw_perturb": [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            "use_yaw": [9, 3],
            # 9 parameters for rotation and 3 parameters for transalation
            "use_rectify": [12, 3],
        }

        if self.opts.model_name == "posecnn":
            self.models["pose"] = model.PoseCNN(
                num_input_frames=2,
                n_out=self.num_parameters[self.opts.pose_repr][0] + self.num_parameters[self.opts.pose_repr][1],
            )
        elif self.opts.model_name == "siamese":
            self.models["pose"] = model.RotationNet(
                n_out=self.num_parameters[self.opts.pose_repr][0],
                use_perturb=self.opts.pose_repr in ["use_perturb"],
                in_channels=5 if self.opts.use_opt_flow_input else 3,
            )
        elif self.opts.model_name == "siamese_deep":
            self.models["pose"] = model.RotationNetDeep(n_out=self.num_parameters[self.opts.pose_repr][0])

        elif self.opts.model_name == "hsm":
            self.models["pose"] = model.HSMNet(
                n_out=self.num_parameters[self.opts.pose_repr], use_tanh=False, dataset=self.opts.dataset
            )

        elif self.opts.model_name == "gmflow":
            self.models["pose"] = model.GMFlow(
                n_out=self.num_parameters[self.opts.pose_repr],
                dataset=self.opts.dataset,
                ablate_init=self.opts.ablate_init,
                ablate_transformer=self.opts.ablate_transformer,
                ablate_volume=self.opts.ablate_volume,
                ablate_res=self.opts.ablate_res,
            )

        if self.opts.use_dp:
            self.models["pose"] = nn.DataParallel(
                self.models["pose"], device_ids=[i for i in range(self.gpu_count - 1)]
            )
            self.models["pose"].cuda()
        else:
            self.models["pose"].to(self.device)

        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opts.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opts.scheduler_step_size, 0.7)

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
                "hd1k": "Raft_Large_Weights.C_T_SKHT_V1",
            }

            self.opt_flow = TorchvisionRAFT(pretrained_weights=self.RAFT_model[self.opts.dataset])
            self.opt_flow.to(self.opt_flow_device)

        if self.opts.eval_mode:
            assert self.opts.load_weights_folder is not None, "Please provide a model checkpoint to run eval mode."
            assert self.opts.batch_size == 1, "In eval mode only batch size 1 is supported until someone figures out how to collate keypoint matches from different samples in a larger batch"

        if self.opts.load_weights_folder is not None:
            self.load_model()

        if self.opts.eval_mode:
            self.superglue = model.SuperGlueMatcher(
                self.opts.nms_radius,
                self.opts.keypoint_threshold,
                self.opts.max_keypoints,
                self.opts.superglue_weights,
                self.opts.sinkhorn_iterations,
                self.opts.match_threshold,
            )
            self.superglue.to(self.device)
            self.superglue.eval()

            self.loftr = model.LoftrMatcher()
            self.loftr.to(self.device)
            self.loftr.eval()

            self.keypoint_models = {"superglue": self.superglue, "loftr": self.loftr}

        self.loss_fn = torch.nn.L1Loss() if self.opts.loss_fn == "l1" else torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineSimilarity()

        print("experiment number:\n  Training model named:\n  ", self.opts.exp_num, self.opts.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opts.log_dir)
        if self.opts.use_dp:
            print("Training is using:\n ", [i for i in range(self.gpu_count)])
        else:
            print("Training is using:\n ", self.device)

        dataset_dict = {
            "torc": dataset.StereoPoseDataset,
            "carla": dataset.CarlaStereoPoseDataset,
            "carla2": dataset.CarlaStereoPose2Dataset,
            "kitti": dataset.KITTIDataset,
        }

        dir_dict = {
            "gt": self.opts.gt_dir,
            "data": self.opts.data_dir,
            "splits": self.opts.splits_dir,
        }

        train_dataset = dataset_dict[self.opts.dataset](
            dir_dict["gt"],
            dir_dict["data"],
            os.path.join(dir_dict["splits"], "train_dataset.csv"),
            self.opts.height,
            self.opts.width,
            True,
            self.opts.pose_repr,
            self.opts.is_crop,
            self.opts.resize_orig_img,
            self.opts.perturbation_level,
        )

        val_dataset = dataset_dict[self.opts.dataset](
            dir_dict["gt"],
            dir_dict["data"],
            os.path.join(dir_dict["splits"], "val_dataset.csv"),
            self.opts.height,
            self.opts.width,
            False,
            self.opts.pose_repr,
            self.opts.is_crop,
            self.opts.resize_orig_img,
            self.opts.perturbation_level,
        )

        self.num_total_steps = len(train_dataset) // self.opts.batch_size * self.opts.num_epochs

        if self.opts.use_subset:
            train_dataset = torch.utils.data.Subset(train_dataset, range(500))
            val_dataset = torch.utils.data.Subset(val_dataset, range(50))
            self.train_loader = DataLoader(
                train_dataset,
                self.opts.batch_size,
                False,
                num_workers=self.opts.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                self.opts.batch_size,
                False,
                num_workers=self.opts.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            self.test_loader = DataLoader(
                val_dataset, self.opts.batch_size, False, num_workers=self.opts.num_workers, pin_memory=True, drop_last=True
            )

        else:
            self.train_loader = DataLoader(
                train_dataset,
                self.opts.batch_size,
                num_workers=self.opts.num_workers,
                pin_memory=False,
                drop_last=True,
                shuffle=True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                self.opts.batch_size,
                num_workers=self.opts.num_workers,
                pin_memory=False,
                drop_last=True,
                shuffle=True,
            )
            self.test_loader = DataLoader(
                val_dataset, 1, num_workers=self.opts.num_workers, pin_memory=False, drop_last=True, shuffle=False
            )

        self.val_iter = iter(self.val_loader)

        # TensorBoard Logging
        self.writers = {}
        for mode in ["train", "val", "test"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        print(f"There are {len(train_dataset):d} training items and {len(val_dataset):d} validation items\n")

        self.save_opts()

    def run_epoch(self):
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            # print(inputs.keys())
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            self.model_optimizer.step()

            duration = time.time() - before_op_time

            log_step = self.step % 100 == 0

            if log_step:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log_rotation_errors(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    def val(self):
        """Validate the model on a single minibatch"""
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

    def run_infer(self):
        """Testing the model on the Validation Set"""

        print("Testing")
        self.set_eval()

        errors = {}
        for _batch_idx, inputs in tqdm(enumerate(self.test_loader)):
            with torch.no_grad():
                outputs, _ = self.process_batch(inputs, mode="test")
                if self.opts.save_imgs:
                    self.save_images(inputs, outputs)
                try:
                    for keypoint_model in self.keypoint_models.keys():
                        self.compute_keypoint_offset(inputs, outputs, errors, keypoint=keypoint_model)
                    self.compute_of_errors(inputs, outputs, errors)
                except IndexError:
                    continue

                self.log_rotation_errors(inputs, outputs, errors)
                self.log("test", inputs, outputs, errors)
            self.step += 1

        average_errors = {}
        writer = self.writers["test"]
        for key, error in errors.items():
            average_errors[key] = [sum(error) / len(error)]
            writer.add_scalar(f"{key}", sum(error) / len(error), self.step)

        df = pd.DataFrame.from_dict(average_errors)
        error_save_path = os.path.join(self.log_path, f"error_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        df.to_csv(error_save_path)
        print(f"Error logs saved to {error_save_path}")

        self.set_train()

    def compute_keypoint_offset(self, inputs, outputs, offsets, keypoint="superglue"):
        """Estimate average keypoint offset in the Y-axis"""
        of_resize = self.opts.algolux_test_res_opt_flow

        inps_dict = {
            "inp": [inputs["full_left"], inputs["full_right"]],
            "gt": [outputs["gt_left_rectf"], outputs["gt_right_rectf"]],
            "pred": [outputs["pred_left_rectf"], outputs["pred_right_rectf"]],
        }

        for key, imgs in inps_dict.items():
            img1 = F.interpolate(imgs[0], scale_factor=(of_resize, of_resize))
            img2 = F.interpolate(imgs[1], scale_factor=(of_resize, of_resize))

            try:
                matches = self.keypoint_models[keypoint](img1, img2)
                offsets[f"{keypoint}_offset/{key}"] = [
                    *offsets.get(f"{keypoint}_offset/{key}", []),
                    torch.abs(matches["mkeypoints0"][:, 1] - matches["mkeypoints1"][:, 1])
                    .mean()
                    .detach()
                    .cpu()
                    .numpy(),
                ]
                outputs[f"{keypoint}_matches/{key}"] = matches
            except IndexError:
                pass

    def compute_of_errors(self, inputs, outputs, errors):
        """Estimate OF error for input, GT and predicted imgs"""
        of_resize = self.opts.algolux_test_res_opt_flow

        inps_dict = {
            "pred": [outputs["pred_left_rectf"], outputs["pred_right_rectf"]],
            "gt": [outputs["gt_left_rectf"], outputs["gt_right_rectf"]],
            "input": [inputs["full_left"], inputs["full_right"]],
        }

        for key, img_list in inps_dict.items():
            img1 = F.interpolate(img_list[0], scale_factor=(of_resize, of_resize))
            img2 = F.interpolate(img_list[1], scale_factor=(of_resize, of_resize))

            # calculate optical flow
            flow_x, flow_y = self.opt_flow(img1.to(self.opt_flow_device), img2.to(self.opt_flow_device))

            flow_x = torch.nan_to_num(flow_x, nan=0.0)

            assert not (torch.any(flow_x.isnan()) or torch.any(flow_y.isnan())), (
                "NaN values detected in optical flow calculation"
            )

            of_loss = torch.abs(flow_y).mean()

            assert not of_loss.isnan(), "optical flow loss is nan"

            errors[f"of_error/{key}"] = [*errors.get(f"of_error/{key}", []), of_loss.detach().cpu().numpy()]
            outputs[f"flow_x/{key}"] = flow_x.detach().cpu().numpy()
            outputs[f"flow_y/{key}"] = flow_y.detach().cpu().numpy()

    def process_batch(self, inputs, mode="train"):
        """Pass a minibatch through the network and generate images and losses"""
        outputs = {}
        for key, ipt in inputs.items():
            if key not in ["img_id", "dataset"]:
                if self.opts.use_dp:
                    inputs[key] = ipt.cuda(non_blocking=True)
                else:
                    inputs[key] = ipt.to(self.device)

        if self.opts.model_name in ["gmflow"]:
            rotation = self.models["pose"](
                inputs["target_aug"],
                inputs["reference_aug"],
                inputs["translation"].unsqueeze(-1),
                inputs["intrns_left"],
                inputs["intrns_right"],
                inputs["dist_coeff1"],
                inputs["dist_coeff2"],
                inputs["full_height"][0],
                inputs["full_width"][0],
            )

            if self.opts.pose_repr in ["use_perturb", "use_identity", "use_yaw_perturb", "use_yaw"]:
                outputs["rotation"] = rotation.reshape(-1, 3, 3) + inputs["rotation_perturb"]
            elif self.opts.pose_repr == ["use_perturb_6D", "use_perturb_quat"]:
                outputs["rotation"] = rotation + inputs["rotation_perturb"]
            elif self.opts.pose_repr == "use_rectify":
                outputs["R1"] = rotation[:, :6]
                outputs["R2"] = rotation[:, 6:]
                outputs["rotation"] = rotation[:, 6:]
            else:
                outputs["rotation"] = rotation

        else:
            raise NotImplementedError

        self.generate_rectf_imgs(inputs, outputs)

        losses = None
        if mode != "test":
            losses = self.compute_losses(inputs, outputs, mode)

        return outputs, losses

    def generate_rectf_imgs(self, inputs, outputs):
        pred_rotation = outputs["rotation"]
        gt_rotation = inputs["rotation"]

        if self.opts.pose_repr in ["use_6D", "use_perturb_6D"]:
            pred_rmat = gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:])
            gt_rmat = gram_schmidt(gt_rotation[:, :3], gt_rotation[:, 3:])
        elif self.opts.pose_repr in ["use_9D", "use_perturb", "use_yaw_perturb", "use_identity", "use_yaw"]:
            pred_rmat = svd_orthogonalize(pred_rotation.reshape(-1, 3, 3))
            gt_rmat = gt_rotation
        elif self.opts.pose_repr in ["use_quat", "use_perturb_quat"]:
            pred_rmat = quaternion_to_matrix(pred_rotation)
            gt_rmat = quaternion_to_matrix(gt_rotation)
        elif self.opts.pose_repr == "use_rectify":
            pred_r1, pred_r2 = (
                gram_schmidt(outputs["R1"][:, :3], outputs["R1"][:, 3:]),
                gram_schmidt(outputs["R2"][:, :3], outputs["R2"][:, 3:]),
            )
            gt_r1, gt_r2 = inputs["R1"], inputs["R2"]
            P1, P2 = inputs["P1"], inputs["P2"]
        else:
            pred_rmat = pred_rotation
            gt_rmat = gt_rotation

        translation = inputs["translation"].unsqueeze(-1)

        intrinsics_left, intrinsics_right = inputs["intrns_left"], inputs["intrns_right"]

        dist1 = inputs["dist_coeff1"]
        dist2 = inputs["dist_coeff2"]

        if self.opts.pose_repr == "use_rectify":
            outputs["pred_left_rectf"], outputs["pred_right_rectf"] = (
                undistortrectify(inputs["full_left"], intrinsics_left, dist1, pred_r1, P1[..., :3]),
                undistortrectify(inputs["full_right"], intrinsics_right, dist2, pred_r2, P2[..., :3]),
            )

            outputs["gt_left_rectf"], outputs["gt_right_rectf"] = (
                undistortrectify(inputs["full_left"], intrinsics_left, dist1, gt_r1, P1[..., :3]),
                undistortrectify(inputs["full_right"], intrinsics_right, dist2, gt_r2, P2[..., :3]),
            )
        else:
            outputs["pred_left_rectf"], outputs["pred_right_rectf"], _ = calibrate_stereo_pair_torch(
                inputs["full_left"],
                inputs["full_right"],
                intrinsics_left,
                intrinsics_right,
                dist1,
                dist2,
                pred_rmat.double(),
                translation.double(),
                inputs["full_height"][0],
                inputs["full_width"][0],
            )

            outputs["gt_left_rectf"], outputs["gt_right_rectf"], _ = calibrate_stereo_pair_torch(
                inputs["full_left"],
                inputs["full_right"],
                intrinsics_left,
                intrinsics_right,
                dist1,
                dist2,
                gt_rmat.double(),
                translation.double(),
                inputs["full_height"][0],
                inputs["full_width"][0],
            )

    def compute_losses(self, inputs, outputs, mode="train"):
        losses = {}
        loss = 0
        gt_rotation = inputs["rotation"]
        pred_rotation = outputs["rotation"]

        of_resize = self.opts.algolux_test_res_opt_flow

        if self.opts.pose_repr in ["use_6D", "use_perturb_6D"]:
            pred_rmat = gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:])
            gt_rmat = gram_schmidt(gt_rotation[:, :3], gt_rotation[:, 3:])
        elif self.opts.pose_repr in ["use_9D", "use_perturb", "use_identity", "use_yaw_perturb", "use_yaw"]:
            pred_rmat = svd_orthogonalize(pred_rotation.reshape(-1, 3, 3))
            gt_rmat = gt_rotation
        elif self.opts.pose_repr in ["use_quat", "use_perturb_quat"]:
            pred_rmat = quaternion_to_matrix(pred_rotation)
            gt_rmat = quaternion_to_matrix(gt_rotation)
        elif self.opts.pose_repr == "use_rectify":
            pred_r1, pred_r2 = (
                gram_schmidt(outputs["R1"][:, :3], outputs["R1"][:, 3:]),
                gram_schmidt(outputs["R2"][:, :3], outputs["R2"][:, 3:]),
            )
            gt_r1, gt_r2 = inputs["R1"], inputs["R2"]
            _P1, _P2 = inputs["P1"], inputs["P2"]
        else:
            pred_rmat = pred_rotation
            gt_rmat = gt_rotation

        if self.opts.pose_repr == "use_rectify":
            losses["rotation"] = self.loss_fn(gt_r1.double(), pred_r1.double()) + self.loss_fn(
                gt_r2.double(), pred_r2.double()
            )
        else:
            losses["rotation"] = self.loss_fn(
                matrix_to_euler_angles(gt_rmat.double(), "XYZ"), matrix_to_euler_angles(pred_rmat.double(), "XYZ")
            )

        loss += self.opts.rotation * losses["rotation"]

        if self.opts.y_axis_euler_loss and self.opts.pose_repr != "use_rectify":
            losses["y_axis"] = self.loss_fn(
                matrix_to_euler_angles(gt_rmat.double(), "XYZ")[..., 1:2],
                matrix_to_euler_angles(pred_rmat.double(), "XYZ")[..., 1:2],
            )
            loss += self.opts.rotation * losses["y_axis"] * 2

        if self.opts.use_opt_flow_loss and self.epoch >= self.opts.epoch_visual_loss:
            of_height, _of_width = self.opts.height, self.opts.width

            if self.opts.use_opt_flow_random_crop:
                # If error line below is broken fix issue if using random crops Width and full_width are the same
                x, y = (
                    torch.randint(0, inputs["full_width"][0] - self.opts.width, (1,)),
                    torch.randint(0, inputs["full_height"][0] - self.opts.height, (1,)),
                )

                img1 = outputs["pred_left_rectf"][:, :, y : y + self.opts.height, x : x + self.opts.width]
                img2 = outputs["pred_right_rectf"][:, :, y : y + self.opts.height, x : x + self.opts.width]
            else:
                # downsample the image for calculation of optical flow
                img1 = F.interpolate(outputs["pred_left_rectf"], scale_factor=(of_resize, of_resize))
                img2 = F.interpolate(outputs["pred_right_rectf"], scale_factor=(of_resize, of_resize))
                of_height, _of_width = img1.shape[2], img1.shape[3]

            # create joint mask of ROI for left and right images and downsample it
            joint_mask = inputs["mask_left"] & inputs["mask_right"]
            if self.opts.use_opt_flow_random_crop:
                joint_mask = joint_mask.double().unsqueeze(1)[:, :, y : y + self.opts.height, x : x + self.opts.width]
            else:
                joint_mask = F.interpolate(joint_mask.double().unsqueeze(1), scale_factor=(of_resize, of_resize))

            # calculate optical flow
            flow_x, flow_y = self.opt_flow(img1.to(self.opt_flow_device), img2.to(self.opt_flow_device))
            if self.opts.use_sky_mask:
                flow_x, flow_y = flow_x[:, of_height // 3 :, :], flow_y[:, of_height // 3 :, :]

            flow_x = torch.nan_to_num(flow_x, nan=0.0)

            assert not (torch.any(flow_x.isnan()) or torch.any(flow_y.isnan())), (
                "NaN values detected in optical flow calculation"
            )

            if self.opts.use_radial_flow_mask:
                radial_mask_full = inputs["radial_mask_full"].to(self.opt_flow_device)

                if self.opts.use_opt_flow_random_crop:
                    radial_mask = radial_mask_full[:, :, y : y + self.opts.height, x : x + self.opts.width]
                else:
                    radial_mask = F.interpolate(radial_mask_full, scale_factor=(of_resize, of_resize))
                    if self.opts.use_sky_mask:
                        radial_mask = radial_mask[:, :, of_height // 3 :, :]

                inputs["radial_mask"] = radial_mask
                losses["opt_flow"] = (radial_mask * torch.abs(flow_y)).mean()
            else:
                losses["opt_flow"] = torch.abs(flow_y).mean()

            losses["negative_opt_flow"] = torch.count_nonzero(flow_x > 0) / (
                inputs["full_height"][0] * inputs["full_width"][0]
            ).to(flow_x.device)

            assert not losses["opt_flow"].isnan(), "optical flow loss is nan"
            loss += (losses["opt_flow"] / 10).to(self.device)

            if not losses["negative_opt_flow"].isnan():
                loss += losses["negative_opt_flow"].to(self.device) * 10

            # store optical flow for logging
            outputs["flow_x"] = flow_x.detach().cpu().numpy()
            outputs["flow_y"] = flow_y.detach().cpu().numpy()
            outputs["joint_mask"] = joint_mask.cpu().numpy()

        if self.opts.constrain_roi:
            black_pixels_left = outputs["pred_left_rectf"].mean(dim=1) == 0
            black_pixels_right = outputs["pred_right_rectf"].mean(dim=1) == 0
            black_pixels = torch.count_nonzero(black_pixels_left) + torch.count_nonzero(black_pixels_right)

            losses["black_pixel_count"] = black_pixels / (inputs["full_height"][0] * inputs["full_width"][0]).to(
                black_pixels.device
            )

            loss += losses["black_pixel_count"]

        losses["loss"] = loss

        return losses

    def compute_rotation_errors(self, pred_rotation, gt_axisangles):
        if self.opts.pose_repr == "use_3D":
            pred_axisangles = pred_rotation.numpy().reshape(-1, 3)
        elif self.opts.pose_repr in ["use_quat", "use_perturb_quat"]:
            pred_rotation = pred_rotation.numpy().reshape(-1, 4)
            pred_axisangles = Rotation.from_quat(pred_rotation).as_rotvec(degrees=True).astype(np.float32)
        elif self.opts.pose_repr == "use_6D":
            pred_rotation = pred_rotation.reshape(-1, 6)
            pred_axisangles = (
                Rotation.from_matrix(gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:]).numpy())
                .as_rotvec(degrees=True)
                .astype(np.float32)
            )
        elif self.opts.pose_repr == "use_9D":
            pred_rotation = pred_rotation.reshape(-1, 3, 3)
            pred_axisangles = (
                Rotation.from_matrix(svd_orthogonalize(pred_rotation).numpy())
                .as_rotvec(degrees=True)
                .astype(np.float32)
            )
        elif self.opts.pose_repr in ["use_perturb", "use_yaw_perturb", "use_identity", "use_yaw"]:
            pred_rotation = pred_rotation
            pred_axisangles = (
                Rotation.from_matrix(svd_orthogonalize(pred_rotation).numpy())
                .as_rotvec(degrees=True)
                .astype(np.float32)
            )
        elif self.opts.pose_repr == "use_perturb_6D":
            pred_rotation = pred_rotation.reshape(-1, 6)
            pred_axisangles = (
                Rotation.from_matrix(gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:]).numpy())
                .as_rotvec(degrees=True)
                .astype(np.float32)
            )

        delta = np.absolute(gt_axisangles - pred_axisangles)

        return delta

    def log_rotation_errors(self, inputs, outputs, losses):
        gt_axisangles = inputs["axisangles"].cpu().numpy()
        pred_rotation = outputs["rotation"].detach().cpu()

        if self.opts.pose_repr == "use_rectify":
            pred_r1, pred_r2 = outputs["R1"], outputs["R2"]
            gt_r1, gt_r2 = inputs["R1"], inputs["R2"]
            _P1, _P2 = inputs["P1"], inputs["P2"]
            pred1_axisangles = (
                Rotation.from_matrix(gram_schmidt(pred_r1[:, :3], pred_r1[:, 3:]).detach().cpu().numpy())
                .as_rotvec(degrees=True)
                .astype(np.float32)
            )
            pred2_axisangles = (
                Rotation.from_matrix(gram_schmidt(pred_r2[:, :3], pred_r2[:, 3:]).detach().cpu().numpy())
                .as_rotvec(degrees=True)
                .astype(np.float32)
            )
            gt1_axisangles = Rotation.from_matrix(gt_r1.cpu().numpy()).as_rotvec(degrees=True).astype(np.float32)
            gt2_axisangles = Rotation.from_matrix(gt_r2.cpu().numpy()).as_rotvec(degrees=True).astype(np.float32)

            delta1 = np.absolute(gt1_axisangles - pred1_axisangles)
            delta2 = np.absolute(gt2_axisangles - pred2_axisangles)

            losses["axisangle_degrees/delta1_x"] = [*losses.get("axisangle_degrees/delta1_x", []), delta1[:, 0].mean()]
            losses["axisangle_degrees/delta1_y"] = [*losses.get("axisangle_degrees/delta1_y", []), delta1[:, 1].mean()]
            losses["axisangle_degrees/delta1_z"] = [*losses.get("axisangle_degrees/delta1_z", []), delta1[:, 2].mean()]
            losses["axisangle_degrees/delta2_x"] = [*losses.get("axisangle_degrees/delta2_x", []), delta2[:, 0].mean()]
            losses["axisangle_degrees/delta2_y"] = [*losses.get("axisangle_degrees/delta2_y", []), delta2[:, 1].mean()]
            losses["axisangle_degrees/delta2_z"] = [*losses.get("axisangle_degrees/delta2_z", []), delta2[:, 2].mean()]

        else:
            delta = self.compute_rotation_errors(pred_rotation, gt_axisangles)

            losses["axisangle_degrees/delta_x"] = [*losses.get("axisangle_degrees/delta_x", []), delta[:, 0].mean()]
            losses["axisangle_degrees/delta_y"] = [*losses.get("axisangle_degrees/delta_y", []), delta[:, 1].mean()]
            losses["axisangle_degrees/delta_z"] = [*losses.get("axisangle_degrees/delta_z", []), delta[:, 2].mean()]

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        writer = self.writers[mode]
        for loss_label, v in losses.items():
            if mode in ["test"]:
                writer.add_scalar(f"{loss_label}", sum(v) / len(v), self.step) # In-efficient moving average
            else:
                writer.add_scalar(f"{loss_label}", v, self.step)

        # Generate rectified images from both ground truth and predicted poses
        pred_rotation = outputs["rotation"].detach().cpu()
        gt_rotation = inputs["rotation"].detach().cpu()
        if self.opts.pose_repr in [
            "use_perturb",
            "use_perturb_quat",
            "use_perturb_6D",
            "use_identity",
            "use_yaw_perturb",
        ]:
            init_rotation = inputs["rotation_perturb"].detach().cpu()

        if self.opts.pose_repr in ["use_6D", "use_perturb_6D"]:
            pred_rmat = (
                Rotation.from_matrix(gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:]).numpy())
                .as_matrix()
                .astype(np.float32)
            )
            gt_rmat = (
                Rotation.from_matrix(gram_schmidt(gt_rotation[:, :3], gt_rotation[:, 3:]).numpy())
                .as_matrix()
                .astype(np.float32)
            )
            if self.opts.pose_repr == "use_perturb_6D":
                init_rmat = (
                    Rotation.from_matrix(gram_schmidt(init_rotation[:, :3], init_rotation[:, 3:]).numpy())
                    .as_matrix()
                    .astype(np.float32)
                )

        elif self.opts.pose_repr in ["use_9D", "use_perturb", "use_identity", "use_yaw_perturb", "use_yaw"]:
            pred_rotation = pred_rotation.reshape(-1, 3, 3)
            pred_rmat = svd_orthogonalize(pred_rotation).numpy().astype(np.float32)
            gt_rmat = gt_rotation.numpy().astype(np.float32)
            if self.opts.pose_repr in ["use_perturb", "use_identity", "use_yaw_perturb"]:
                init_rmat = init_rotation.numpy().astype(np.float32)

        elif self.opts.pose_repr in ["use_quat", "use_perturb_quat"]:
            pred_rotation = pred_rotation.numpy().reshape(-1, 4)
            pred_rmat = Rotation.from_quat(pred_rotation).as_matrix().astype(np.float32)
            gt_rotation = gt_rotation.numpy().reshape(-1, 4)
            gt_rmat = Rotation.from_quat(gt_rotation).as_matrix().astype(np.float32)
            if self.opts.pose_repr == "use_perturb_quat":
                init_rotation = init_rotation.numpy().reshape(-1, 4)
                init_rmat = Rotation.from_quat(init_rotation).as_matrix().astype(np.float32)

        elif self.opts.pose_repr == "use_rectify":
            pred_r1, pred_r2 = (
                gram_schmidt(outputs["R1"][:, :3], outputs["R1"][:, 3:]).detach().cpu().numpy(),
                gram_schmidt(outputs["R2"][:, :3], outputs["R2"][:, 3:]).detach().cpu().numpy(),
            )
            gt_r1, gt_r2 = inputs["R1"].cpu().numpy(), inputs["R2"].cpu().numpy()
            P1, P2 = inputs["P1"].cpu().numpy(), inputs["P2"].cpu().numpy()

        translation = inputs["translation"].detach().cpu().numpy()

        left_imgs = inputs["full_left"].permute(0, 2, 3, 1).cpu().numpy()
        right_imgs = inputs["full_right"].permute(0, 2, 3, 1).cpu().numpy()
        for j in range(min(4, self.opts.batch_size)):
            unrect_left_img = (left_imgs[j, ...] * 255.0).astype(np.uint8)
            unrect_right_img = (right_imgs[j, ...] * 255.0).astype(np.uint8)

            height, width, _ = unrect_left_img.shape

            if self.opts.pose_repr != "use_rectify":
                R_pred = pred_rmat[j, ...]
                R_gt = gt_rmat[j, ...]

            t = translation[j, ...]

            K_left, K_right = inputs["intrns_left"][j].cpu().numpy(), inputs["intrns_right"][j].cpu().numpy()

            dist1 = inputs["dist_coeff1"][j].cpu().numpy()
            dist2 = inputs["dist_coeff2"][j].cpu().numpy()

            if self.opts.pose_repr == "use_rectify":
                pred_img_left_rect, pred_img_right_rect = undistortrectify_cv2(
                    unrect_left_img,
                    unrect_right_img,
                    K_left.astype(np.float64),
                    K_right.astype(np.float64),
                    dist1.astype(np.float64),
                    dist2.astype(np.float64),
                    pred_r1[j, ...].astype(np.float64),
                    pred_r2[j, ...].astype(np.float64),
                    P1[j, ...].astype(np.float64),
                    P2[j, ...].astype(np.float64),
                    height,
                    width,
                )
                gt_img_left_rect, gt_img_right_rect = undistortrectify_cv2(
                    unrect_left_img,
                    unrect_right_img,
                    K_left.astype(np.float64),
                    K_right.astype(np.float64),
                    dist1.astype(np.float64),
                    dist2.astype(np.float64),
                    gt_r1[j, ...].astype(np.float64),
                    gt_r2[j, ...].astype(np.float64),
                    P1[j, ...].astype(np.float64),
                    P2[j, ...].astype(np.float64),
                    height,
                    width,
                )

            else:
                pred_img_left_rect, pred_img_right_rect, _ = calibrate_stereo_pair(
                    unrect_left_img,
                    unrect_right_img,
                    K_left.astype(np.float64),
                    K_right.astype(np.float64),
                    dist1.astype(np.float64),
                    dist2.astype(np.float64),
                    R_pred.astype(np.float64),
                    t.astype(np.float64),
                    height,
                    width,
                )

                gt_img_left_rect, gt_img_right_rect, _ = calibrate_stereo_pair(
                    unrect_left_img,
                    unrect_right_img,
                    K_left.astype(np.float64),
                    K_right.astype(np.float64),
                    dist1.astype(np.float64),
                    dist2.astype(np.float64),
                    R_gt.astype(np.float64),
                    t.astype(np.float64),
                    height,
                    width,
                )

            pred_ep_image_pair = draw_epipolar_lines(
                pred_img_left_rect, pred_img_right_rect, num_lines=50, line_color=(0, 0, 0), text=inputs["img_id"][j]
            )
            gt_ep_image_pair = draw_epipolar_lines(
                gt_img_left_rect, gt_img_right_rect, num_lines=50, line_color=(0, 0, 0), text=inputs["img_id"][j]
            )
            input_ep_image_pair = draw_epipolar_lines(
                unrect_left_img.copy(),
                unrect_right_img.copy(),
                num_lines=50,
                line_color=(0, 0, 0),
                text=inputs["img_id"][j],
            )

            pred_image_overlay = cv2.addWeighted(pred_img_left_rect, 0.5, pred_img_right_rect, 0.5, 0)

            gt_image_overlay = cv2.addWeighted(gt_img_left_rect, 0.5, gt_img_right_rect, 0.5, 0)

            input_image_overlay = cv2.addWeighted(unrect_left_img, 0.5, unrect_right_img, 0.5, 0)

            ### Visualize initialization
            if self.opts.pose_repr in [
                "use_perturb",
                "use_perturb_quat",
                "use_perturb_6D",
                "use_identity",
                "use_yaw_perturb",
            ]:
                R_init = init_rmat[j, ...]

                init_img_left_rect, init_img_right_rect, _ = calibrate_stereo_pair(
                    unrect_left_img,
                    unrect_right_img,
                    K_left.astype(np.float64),
                    K_right.astype(np.float64),
                    dist1.astype(np.float64),
                    dist2.astype(np.float64),
                    R_init.astype(np.float64),
                    t.astype(np.float64),
                    height,
                    width,
                )

                init_ep_image_pair = draw_epipolar_lines(
                    init_img_left_rect,
                    init_img_right_rect,
                    num_lines=50,
                    line_color=(0, 0, 0),
                    text=inputs["img_id"][j],
                )

                init_image_overlay = cv2.addWeighted(init_img_left_rect, 0.5, init_img_right_rect, 0.5, 0)

                writer.add_image(f"init_epipolar_images/{j}", init_ep_image_pair, self.step, dataformats="HWC")
                writer.add_image(f"init_overlay/{j}", init_image_overlay, self.step, dataformats="HWC")

            scale_percent = 50  # percent of original size
            width = int(pred_ep_image_pair.shape[1] * scale_percent / 100)
            height = int(pred_ep_image_pair.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            pred_ep_image_pair = cv2.resize(pred_ep_image_pair, dim, interpolation=cv2.INTER_AREA)
            gt_ep_image_pair = cv2.resize(gt_ep_image_pair, dim, interpolation=cv2.INTER_AREA)

            writer.add_image(f"input_epipolar_images/{j}", input_ep_image_pair, self.step, dataformats="HWC")
            writer.add_image(f"pred_epipolar_images/{j}", pred_ep_image_pair, self.step, dataformats="HWC")
            writer.add_image(f"gt_epipolar_images/{j}", gt_ep_image_pair, self.step, dataformats="HWC")
            writer.add_image(f"input_overlay/{j}", input_image_overlay, self.step, dataformats="HWC")
            writer.add_image(f"pred_overlay/{j}", pred_image_overlay, self.step, dataformats="HWC")
            writer.add_image(f"gt_overlay/{j}", gt_image_overlay, self.step, dataformats="HWC")

            if mode == "test":
                for key in ["inp", "gt", "pred"]:
                    if f"flow_x/{key}" in outputs:
                        flow_x = np.clip(outputs[f"flow_x/{key}"][j], 0.0, 255.0)
                        flow_y = np.clip(outputs[f"flow_y/{key}"][j], 0.0, 255.0)
                        f = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
                        opt_flow_img = flow_to_image(f)
                        writer.add_image(f"optical_flow/{key}_{j}", opt_flow_img, self.step, dataformats="HWC")

            elif self.opts.use_opt_flow_loss and (self.epoch >= self.opts.epoch_visual_loss):
                # optical flow visualization
                flow_x = np.clip(outputs["flow_x"][j], 0.0, 255.0)
                flow_y = np.clip(outputs["flow_y"][j], 0.0, 255.0)
                f = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
                opt_flow_img = flow_to_image(f)
                writer.add_image(f"optical_flow/{j}", opt_flow_img, self.step, dataformats="HWC")

                # show ROI mask
                joint_mask = outputs["joint_mask"][j]
                writer.add_image(f"ROI mask/{j}", joint_mask, self.step, dataformats="CHW")

                # show Radial Mask
                if self.opts.use_radial_flow_mask:
                    radial_mask = inputs["radial_mask"].to(self.opt_flow_device)[0]
                    writer.add_image("Radial mask", radial_mask / radial_mask.max(), self.step, dataformats="CHW")

    def save_images(self, inputs, outputs):
        # Generate rectified images from both ground truth and predicted poses
        pred_rotation = outputs["rotation"].detach().cpu()
        gt_rotation = inputs["rotation"].detach().cpu()

        if self.opts.pose_repr in [
            "use_perturb",
            "use_perturb_quat",
            "use_perturb_6D",
            "use_identity",
            "use_yaw_perturb",
        ]:
            init_rotation = inputs["rotation_perturb"].detach().cpu()

        if self.opts.pose_repr in ["use_6D", "use_perturb_6D"]:
            pred_rmat = (
                Rotation.from_matrix(gram_schmidt(pred_rotation[:, :3], pred_rotation[:, 3:]).numpy())
                .as_matrix()
                .astype(np.float32)
            )
            gt_rmat = (
                Rotation.from_matrix(gram_schmidt(gt_rotation[:, :3], gt_rotation[:, 3:]).numpy())
                .as_matrix()
                .astype(np.float32)
            )

            if self.opts.pose_repr == "use_perturb_6D":
                Rotation.from_matrix(
                    gram_schmidt(init_rotation[:, :3], init_rotation[:, 3:]).numpy()
                ).as_matrix().astype(np.float32)

        elif self.opts.pose_repr in ["use_9D", "use_perturb", "use_identity", "use_yaw_perturb", "use_yaw"]:
            pred_rotation = pred_rotation.reshape(-1, 3, 3)
            pred_rmat = svd_orthogonalize(pred_rotation).numpy().astype(np.float32)
            gt_rmat = gt_rotation.numpy().astype(np.float32)

        translation = inputs["translation"].detach().cpu().numpy()

        left_imgs = inputs["full_left"].permute(0, 2, 3, 1).cpu().numpy()
        right_imgs = inputs["full_right"].permute(0, 2, 3, 1).cpu().numpy()

        for j in range(min(4, self.opts.batch_size)):
            unrect_left_img = (left_imgs[j, ...] * 255.0).astype(np.uint8)
            unrect_right_img = (right_imgs[j, ...] * 255.0).astype(np.uint8)

            height, width, _ = unrect_left_img.shape

            if self.opts.pose_repr != "use_rectify":
                R_pred = pred_rmat[j, ...]
                R_gt = gt_rmat[j, ...]

            t = translation[j, ...]

            K_left, K_right = inputs["intrns_left"][j].cpu().numpy(), inputs["intrns_right"][j].cpu().numpy()

            dist1 = inputs["dist_coeff1"][j].cpu().numpy()
            dist2 = inputs["dist_coeff2"][j].cpu().numpy()

            pred_img_left_rect, pred_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img,
                unrect_right_img,
                K_left.astype(np.float64),
                K_right.astype(np.float64),
                dist1.astype(np.float64),
                dist2.astype(np.float64),
                R_pred.astype(np.float64),
                t.astype(np.float64),
                height,
                width,
            )

            gt_img_left_rect, gt_img_right_rect, _ = calibrate_stereo_pair(
                unrect_left_img,
                unrect_right_img,
                K_left.astype(np.float64),
                K_right.astype(np.float64),
                dist1.astype(np.float64),
                dist2.astype(np.float64),
                R_gt.astype(np.float64),
                t.astype(np.float64),
                height,
                width,
            )

            images_dict = {
                "pred": [pred_img_left_rect, pred_img_right_rect, R_pred],
                "gt": [gt_img_left_rect, gt_img_right_rect, R_gt],
                "inp": [unrect_left_img, unrect_right_img, np.eye(3)],
            }

            for method, imgs in images_dict.items():
                save_dir = os.path.join(self.log_path, "saved_imgs", method)
                if not os.path.isdir(save_dir):
                    os.makedirs(os.path.join(save_dir, "left"))
                    os.makedirs(os.path.join(save_dir, "right"))
                    os.makedirs(os.path.join(save_dir, "extrinscs"))
                if not os.path.isdir(os.path.join(save_dir, "intrinsics", "left")):
                    os.makedirs(os.path.join(save_dir, "intrinsics", "left"))
                    os.makedirs(os.path.join(save_dir, "intrinsics", "right"))

                id = inputs["img_id"][j]
                cv2.imwrite(os.path.join(save_dir, "left", f"{id}.png"), imgs[0][:, :, ::-1])
                cv2.imwrite(os.path.join(save_dir, "right", f"{id}.png"), imgs[1][:, :, ::-1])
                T = np.eye(4)
                T[:3, :3] = imgs[-1]
                T[:3, 3] = t
                np.save(os.path.join(save_dir, "extrinscs", f"{id}.npy"), T)
                np.save(os.path.join(save_dir, "intrinsics", "left", f"{id}.npy"), K_left.astype(np.float64))
                np.save(os.path.join(save_dir, "intrinsics", "right", f"{id}.npy"), K_right.astype(np.float64))
