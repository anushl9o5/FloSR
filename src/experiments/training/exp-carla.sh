#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1
python src/train.py \
	--dataset carla \
	--data_dir /path/to/parent/dataset_folder/ \
	--gt_dir /path/to/parent/dataset_folder/camera_poses/ \
	--splits_dir /path/to/parent/dataset_folder/splits/ \
	--log_dir /path/to/log_directory/ \
	--height 512 \
	--width 1024 \
	--exp_num 99 \
	--exp_name training \
	--exp_metainfo carla_6D_gmflow \
	--batch_size 2 \
	--num_workers 12 \
	--pose_repr use_6D \
	--rotation 1e+02 \
	--learning_rate 2e-05 \
	--model_name gmflow \
	--num_epochs 160 \
	--algolux_test_res_opt_flow 0.5 \
	--constrain_roi \
	--epoch_visual_loss 0 \
	--use_radial_flow_mask \
	--resize_orig_img \
	--use_superglue_eval \
	--use_opt_flow_loss \
	--multi_gpu_opt_flow
