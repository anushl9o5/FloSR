#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1
python src/train.py --data_dir              /nas/EOS/users/aman/data/RelativePose/frames \
                    --gt_dir                /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --splits_dir            src/data/split_files/algolux \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                544 \
                    --width                 960 \
                    --exp_num               40 \
                    --exp_name              perturbation_pose_estimation \
                    --exp_metainfo          sterepose_9D_lr_2e-6_perturb_0.5_optflow_544x960 \
                    --batch_size            1 \
                    --num_workers           2 \
		    --pose_repr		    use_perturb \
                    --rotation              1e+02 \
                    --learning_rate         2e-06 \
                    --model_name            siamese \
                    --num_epochs            80 \
                    --test_res_opt_flow     0.3334 \
                    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --use_radial_flow_mask  \
		    --constrain_roi         \
		    --resize_orig_img       \
		    --perturbation_level    0.5 \
