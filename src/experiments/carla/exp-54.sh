#!/bin/sh
export CUDA_VISIBLE_DEVICES=5
python src/train.py --data_dir              /nas/EOS/users/aman/data/RelativePose/frames \
                    --gt_dir                /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --splits_dir            src/data/split_files/algolux \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                576 \
                    --width                 768 \
                    --exp_num               54 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          sterepose_9D_lr_2e-5_perturb_0.2_costvol_576x768 \
                    --batch_size            10 \
                    --num_workers           4 \
		    --pose_repr		    use_perturb \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --test_res_opt_flow     0.3334 \
		    --use_opt_flow_loss     \
		    --use_radial_flow_mask  \
		    --constrain_roi         \
		    --perturbation_level    0.0 \
		    --resize_orig_img       \
		    --use_cost_vol_loss     \
		    --y_axis_euler_loss     \
