#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5
python src/train.py \
	            --algolux_data_dir      /nas/EOS/users/aman/data/RelativePose/frames \
                    --algolux_gt_dir        /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --algolux_splits_dir    src/data/split_files/algolux \
		    --carla_data_dir        /nas/EOS/users/aman/data/RelativePose/frames \
	            --carla_gt_dir          /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
                    --carla_splits_dir      src/data/split_files/algolux \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                576 \
                    --width                 768 \
                    --exp_num               60 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          sterepose_yaw_lr_2e-5_perturb_0.5_iterative_576x768 \
                    --batch_size            10 \
                    --num_workers           4 \
		    --pose_repr		    use_yaw_perturb \
                    --rotation              1e+02 \
                    --learning_rate         1e-05 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --algolux_test_res_opt_flow     0.3334 \
		    --use_opt_flow_loss     \
		    --constrain_roi         \
		    --perturbation_level    0.5 \
		    --resize_orig_img       \
		    --y_axis_euler_loss     \
		    --epoch_visual_loss     1 \
		    --use_sky_mask          \
