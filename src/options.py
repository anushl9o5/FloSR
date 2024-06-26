from __future__ import absolute_import, division, print_function

import os
import argparse

class StereoPoseOptions:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description='Relative camera pose estimation options')

        # PATH options
        self.parser.add_argument("--dataset",
                                    type=str,
                                    default="algolux",
                                    choices=["algolux","carla", "carla2", "conti", "kitti", "argo", "argo2", "dstereo", "torc", "flyingthings", "sintel", "hd1k"],
                                    help="root directory to the dataset")
        
        self.parser.add_argument("--data_dir",
                                    type=str,
                                    required=True,
                                    help="root directory to the dataset")
        self.parser.add_argument("--gt_dir",
                                    type=str,
                                    required=True,
                                    help="root directory to the groundtruth")
        self.parser.add_argument("--splits_dir",
                                    type=str,
                                    required=True,
                                    help="directory with split files")

        self.parser.add_argument("--log_dir",
                                    type=str,
                                    required=True,
                                    help="directory to store logs")

        # ABLATION options
        self.parser.add_argument("--rotation",
                                    type=float,
                                    help="weight for rotation loss",
                                    default=1.0)
        self.parser.add_argument("--pose_repr",
                                  choices = ['use_3D','use_quat','use_6D', 'use_9D', 'use_perturb', 'use_perturb_6D', 'use_perturb_quat', 'use_identity', 'use_yaw', 'use_rectify'],
                                  default='use_6D',  
                                  help="which representation to choose for rotation matrix SO(3)")
        self.parser.add_argument("--is_crop",
                                  action="store_true",
                                  help="whether to crop the image to the given dimensions or to resize to those dimensions")
        self.parser.add_argument("--resize_orig_img",
                                  action="store_true",
                                  help="whether to resize original image itself to fit model on the GPU (different from resize for Optical Flow)")
        self.parser.add_argument("--use_identity_init",
                                  action="store_true",
                                  help="If True, will use Identity matrix as initialization for Rotation correction")
        self.parser.add_argument("--use_superglue_eval",
                                  action="store_true",
                                  help="If True, will use superglue to evaluate on the validation set")
        self.parser.add_argument("--ablate_transformer",
                                  action="store_true",
                                  help="If True, will not use feature enhancer transfomer")
        self.parser.add_argument("--ablate_init",
                                  action="store_true",
                                  help="If True, will not rectify to identity")
        self.parser.add_argument("--ablate_volume",
                                  action="store_true",
                                  help="If True, will not use cost volume and decoder")
        self.parser.add_argument("--ablate_res",
                                  action="store_true",
                                  help="If True, will use lower res inputs")
        self.parser.add_argument("--eval_sota",
                                  action="store_true",
                                  help="If True, will evaluate sota results too")
        # TRAINING options
        self.parser.add_argument("--exp_num",
                                    type=int,
                                    help="experiment number",
                                    default=-1)
        self.parser.add_argument("--exp_name",
                                    type=str,
                                    help="the name of the folder to save the model in",
                                    default="gated2gated")
        self.parser.add_argument("--exp_metainfo",
                                    type=str,
                                    default="Main Experiment",
                                    help="additional info regarding experiment")
        self.parser.add_argument("--height",
                                    type=int,
                                    default=256,
                                    help="crop height of the image")
        self.parser.add_argument("--width",
                                    type=int,
                                    default=512,
                                    help="crop width of the image")
        self.parser.add_argument("--model_name",
                                    default="posecnn",
                                    choices=["posecnn","siamese","siamese_deep","hsm","hsm_iterative", "hsm_gru", "gmflow"],
                                    help="name of model architecture")
        self.parser.add_argument("--loss_fn",
                                    default="l1",
                                    choices=["l1","l2"],
                                    help="type of loss function")
        self.parser.add_argument("--use_dir_loss",
                                    action="store_true",
                                    help="Whether to use direction loss (negative cosine similarity loss)")
        self.parser.add_argument("--use_opt_flow_loss",
                                    action="store_true",
                                    help="Whether to use optical flow loss")
        self.parser.add_argument("--use_visual_loss",
                                    action="store_true",
                                    help="Whether to use visual loss")
        self.parser.add_argument("--use_visual_loss_self",
                                    action="store_true",
                                    help="Whether to use self-supervised visual loss")
        self.parser.add_argument("--epoch_visual_loss",
                                    default=1,
                                    type=int,
                                    help="start epoch to take optical loss or visual loss into account")
        self.parser.add_argument("--hsm_iterations",
                                    default=1,
                                    type=int,
                                    help="No. of iterative corrections to apply")
        self.parser.add_argument("--algolux_test_res_opt_flow",
                                    type=float,
                                    default=0.4,
                                    help="Which resolution to calculate optical flow on Algolux")
        self.parser.add_argument("--carla_test_res_opt_flow",
                                    type=float,
                                    default=0.4,
                                    help="Which resolution to calculate optical flow on Carla")
        self.parser.add_argument("--use_radial_flow_mask",
                                    action="store_true",
                                    help="Whether to use radial mask for optical flow")
        self.parser.add_argument("--use_subset",
                                    action="store_true",
                                    help="Train on a smaller subset of the dataset (quick evaluation purposes only!)")
        self.parser.add_argument("--use_opt_flow_random_crop",
                                    action="store_true",
                                    help="Random Crop for Optical flow loss (maintains original resolution)")
        self.parser.add_argument("--constrain_roi",
                                    action="store_true",
                                    help="Constrain the black pixels in the rectified images (ROI constraint for y-axis)")
        self.parser.add_argument("--use_opt_flow_input",
                                    action="store_true",
                                    help="Calculate and append Optical Flow at both input images")
        self.parser.add_argument("--use_trace_similarity_loss",
                                    action="store_true",
                                    help="Use trace similarity loss b/w rotation matrices to train")
        self.parser.add_argument("--use_epipole_loss",
                                    action="store_true",
                                    help="Use trace similarity loss b/w rotation matrices to train")
        self.parser.add_argument("--y_axis_euler_loss",
                                    action="store_true",
                                    help="add more weightage to error in y-axis")
        self.parser.add_argument("--use_cost_vol_loss",
                                    action="store_true",
                                    help="Minimize dissimilarity between left / right featurs in cost volume")
        self.parser.add_argument("--use_sky_mask",
                                    action="store_true",
                                    help="Mask out top 1/3rd of OF (around sky region)")
        self.parser.add_argument("--use_segmentation_loss",
                                    action="store_true",
                                    help="Use semantic segmentation loss, contrain the y-axis of a region's centroid ")
        self.parser.add_argument("--use_new_costvol_loss",
                                    action="store_true",
                                    help="Feature based rectification loss")
        self.parser.add_argument("--use_dispwarp_feat_loss",
                                    action="store_true",
                                    help="Feature based Disparity warp loss")
        self.parser.add_argument("--use_dispwarp_img_loss",
                                    action="store_true",
                                    help="Image based Disparity warp loss")
        self.parser.add_argument("--all_rotation_loss",
                                    action="store_true",
                                    help="Loss for rotation predictions at all levels")
        self.parser.add_argument("--use_seq_hsm",
                                    action="store_true",
                                    help="Use seq hsm pipeline ")
        self.parser.add_argument("--use_dispfeatvol_loss",
                                    action="store_true",
                                    help="Disparity based vertical cost vol loss ")
        self.parser.add_argument("--use_I_loss",
                                    action="store_true",
                                    help="Make predicted rotation matrix close to Identity matrix")
        # OPTIMIZATION OPTION
        self.parser.add_argument("--batch_size",
                                    type=int,
                                    help="batch size",
                                    default=1)
        self.parser.add_argument("--learning_rate",
                                    type=float,
                                    help="learning rate",
                                    default=1e-4)
        self.parser.add_argument("--start_epoch",
                                    type=int,
                                    help="start epoch to have non-zero starting option for continuing training",
                                    default=0)
        self.parser.add_argument("--num_epochs",
                                    type=int,
                                    help="number of epochs",
                                    default=20)
        self.parser.add_argument("--scheduler_step_size",
                                    type=int,
                                    help="step size of the scheduler",
                                    default=15)  
        self.parser.add_argument("--perturbation_level",
                                    type=float,
                                    help="std of random noise added for rotation",
                                    default=0.2)
        self.parser.add_argument("--add_random_normal",
                                    action="store_true",
                                    help="std of random noise added for stable training")
        self.parser.add_argument("--use_dp",
                                    action="store_true",
                                    help="Use DataParallel training to fit large batches in many small gpus")
        self.parser.add_argument("--save_imgs",
                                    action="store_true",
                                    help="save images in log folder for downstream task inference")
        
        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                    type=int,
                                    help="number of batches between each tensorboard log",
                                    default=250)
        self.parser.add_argument("--chkpt_frequency",
                                    type=int,
                                    help="number of batches between each checkpoint",
                                    default=250)
        self.parser.add_argument("--save_frequency",
                                    type=int,
                                    help="number of epochs between each save",
                                    default=1)
        
        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                    type=str,
                                    help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                    nargs="+",
                                    type=str,
                                    help="models to load",
                                    default=["pose"])
        self.parser.add_argument("--opt_flow_weights",
                                    type=str,
                                    help="path to load weights",
                                    default="/nas/EOS/users/aman/checkpoints/RAFT/raft-things.pth")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                    action="store_true",
                                    help="whether to train on cpu")
        self.parser.add_argument("--num_workers",
                                    type=int,
                                    help="number of dataloader workers",
                                    default=12)
        self.parser.add_argument("--multi_gpu_opt_flow",
                                    action="store_true",
                                    help="whether to load optical flow model on another GPU")

        # SuperPoint + SuperGlue options
        self.parser.add_argument(
            '--superglue_weights', choices={'indoor', 'outdoor'}, default='outdoor',
            help='SuperGlue weights')
        self.parser.add_argument(
            '--max_keypoints', type=int, default=1024,
            help='Maximum number of keypoints detected by Superpoint'
                ' (\'-1\' keeps all keypoints)')
        self.parser.add_argument(
            '--keypoint_threshold', type=float, default=0.005,
            help='SuperPoint keypoint detector confidence threshold')
        self.parser.add_argument(
            '--nms_radius', type=int, default=4,
            help='SuperPoint Non Maximum Suppression (NMS) radius'
            ' (Must be positive)')
        self.parser.add_argument(
            '--sinkhorn_iterations', type=int, default=50,
            help='Number of Sinkhorn iterations performed by SuperGlue')
        self.parser.add_argument(
            '--match_threshold', type=float, default=0.2,
            help='SuperGlue match threshold')



    def parse(self):
            self.options = self.parser.parse_args()
            return self.options