#!/bin/sh
export CUDA_VISIBLE_DEVICES=5
python src/train.py \
	            --dataset               torc \
	            --data_dir              /nas/EOS-DB/users/aman/data/RelativePose/frames \
                    --gt_dir                /nas/EOS-DB/users/aman/data/RelativePose/groundtruth_json_poses \
		    --splits_dir            src/data/split_files/torc/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               101 \
                    --exp_name              gmflow_ablations \
                    --exp_metainfo          torc_6D_gmflow_no_OF \
                    --batch_size            6 \
                    --num_workers           12 \
		    --pose_repr		    use_6D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            gmflow \
                    --num_epochs            160 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     0 \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_superglue_eval    \
		    --y_axis_euler_loss     \
