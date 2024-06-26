#!/bin/sh
export CUDA_VISIBLE_DEVICES=6,7
BASE_DATA_DIR=/nas/EOS-DB/users/aman/data/
python src/train.py \
	            --algolux_data_dir      ${BASE_DATA_DIR}/RelativePose/frames \
                    --algolux_gt_dir        ${BASE_DATA_DIR}/RelativePose/groundtruth_json_poses \
		    --algolux_splits_dir    src/data/split_files/algolux \
		    --carla_data_dir        ${BASE_DATA_DIR}/RelativePose/frames \
	            --carla_gt_dir          ${BASE_DATA_DIR}/RelativePose/groundtruth_json_poses \
                    --carla_splits_dir      src/data/split_files/algolux \
                    --kitti_data_dir        ${BASE_DATA_DIR}/RelativePose/frames \
                    --kitti_gt_dir          ${BASE_DATA_DIR}/RelativePose/groundtruth_json_poses \
                    --kitti_splits_dir      src/data/split_files/kitti \
                    --log_dir               /nas/EOS/users/anush/logs \
		    --dataset               kitti \
                    --height                192 \
                    --width                 640 \
                    --exp_num               71-resume \
                    --exp_name              hsm_debug \
                    --exp_metainfo          kitti_6D_revseq_model_lr_2e-5_192x640 \
                    --batch_size            10 \
                    --num_workers           4 \
		    --pose_repr		    use_9D \
                    --rotation              1e+02 \
                    --learning_rate         1e-05 \
                    --model_name            hsm \
                    --num_epochs            160 \
                    --algolux_test_res_opt_flow     0.5 \
		    --use_opt_flow_loss     \
		    --constrain_roi         \
		    --perturbation_level    0.5 \
		    --epoch_visual_loss     10 \
		    --use_sky_mask          \
		    --use_radial_flow_mask  \
		    --use_visual_loss       \
		    --resize_orig_img       \
		    --use_new_costvol_loss  \
		    --use_seq_hsm           \
		    --use_superglue_eval    \
		    --y_axis_euler_loss     \
