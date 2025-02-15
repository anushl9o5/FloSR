#!/bin/sh
export CUDA_VISIBLE_DEVICES=2,3
python src/train.py \
	            --algolux_data_dir      /nas/EOS/users/aman/data/RelativePose/frames \
                    --algolux_gt_dir        /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --algolux_splits_dir    src/data/split_files/algolux \
		    --carla_data_dir        /nas/EOS/users/aman/data/RelativePose/frames \
	            --carla_gt_dir          /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
                    --carla_splits_dir      src/data/split_files/algolux \
		    --load_weights_folder   /nas/EOS/users/anush/logs/hsm_debug/exp-64_sterepose_6D_new_model_lr_2e-5_576x768/models/weights_0/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                576 \
                    --width                 768 \
                    --exp_num               64 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          sterepose_6D_new_model_lr_2e-5_576x768 \
                    --batch_size            10 \
                    --num_workers           4 \
		    --pose_repr		    use_6D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --algolux_test_res_opt_flow     0.3334 \
		    --use_opt_flow_loss     \
		    --constrain_roi         \
		    --perturbation_level    0.5 \
		    --resize_orig_img       \
		    --y_axis_euler_loss     \
		    --epoch_visual_loss     0 \
		    --use_sky_mask          \
		    --use_segmentation_loss \
		    --use_radial_flow_mask  \
		    --use_visual_loss       \
