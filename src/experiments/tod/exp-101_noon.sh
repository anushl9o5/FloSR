#!/bin/sh
export CUDA_VISIBLE_DEVICES=2,3
python src/infer.py \
	            --dataset               torc \
	            --data_dir              /home/eos/workspace/RelativePose/frames \
                    --gt_dir                /home/eos/workspace/RelativePose/groundtruth_json_poses \
		    --splits_dir            src/data/split_files/torc/noon/ \
                    --log_dir               /home/eos/workspace/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               101 \
                    --exp_name              tod_metrics \
                    --exp_metainfo          torc_6D_gmflow_noon \
                    --batch_size            1 \
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
		    --load_weights_folder   /home/eos/workspace/logs/hsm_debug/exp-101_torc_6D_gmflow/models/weights_21 \
                    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --eval_sota             \

