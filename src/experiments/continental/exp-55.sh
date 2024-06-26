#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1
python src/train.py --data_dir              /nas/EOS/users/aman/data/RelativePose/frames \
                    --gt_dir                /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --splits_dir            src/data/split_files/algolux \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                576 \
                    --width                 768 \
                    --exp_num               55 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          sterepose_quat_lr_2e-6_perturb_0.2_optflow_576x768 \
                    --batch_size            1 \
                    --num_workers           2 \
		    --pose_repr		    use_perturb_quat \
                    --rotation              1e+02 \
                    --learning_rate         2e-06 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --test_res_opt_flow     0.3334 \
                    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --use_radial_flow_mask  \
		    --constrain_roi         \
		    --resize_orig_img       \
		    --perturbation_level    0.2 \
		    --use_epipole_loss      \
		    --add_random_normal     \
