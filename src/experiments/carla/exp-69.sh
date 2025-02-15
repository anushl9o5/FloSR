#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1
python src/train.py \
	            --algolux_data_dir      /nas/EOS/users/aman/data/RelativePose/frames \
                    --algolux_gt_dir        /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --algolux_splits_dir    src/data/split_files/algolux \
		    --carla_data_dir        /nas/EOS/users/aman/data/RelativePose/frames \
	            --carla_gt_dir          /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
                    --carla_splits_dir      src/data/split_files/algolux \
                    --kitti_data_dir        /nas/EOS/users/aman/data/RelativePose/frames \
                    --kitti_gt_dir          /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
                    --kitti_splits_dir      src/data/split_files/kitti \
                    --log_dir               /home/anush/logs \
		    --dataset               kitti \
                    --height                192 \
                    --width                 640 \
                    --exp_num               69 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          kitti_6D_new_model_lr_2e-5_576x768 \
                    --batch_size            5 \
                    --num_workers           4 \
		    --pose_repr		    use_6D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --algolux_test_res_opt_flow     0.5 \
		    --use_opt_flow_loss     \
		    --constrain_roi         \
		    --perturbation_level    0.5 \
		    --y_axis_euler_loss     \
		    --epoch_visual_loss     2 \
		    --use_sky_mask          \
		    --use_radial_flow_mask  \
		    --use_visual_loss       \
		    --resize_orig_img       \
		    --use_superglue_eval    \
		    --use_new_costvol_loss  \
