#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5
python src/train.py \
	            --dataset               argo2 \
	            --data_dir              /nas/EOS/users/anush/argo/ \
                    --gt_dir                /nas/EOS/users/anush/argo/ \
		    --splits_dir            src/data/split_files/argoverse2/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                192 \
                    --width                 640 \
                    --exp_num               86 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          argo2_9D_seq_vertical_costvol \
                    --batch_size            32 \
                    --num_workers           12 \
		    --pose_repr		    use_9D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     5 \
		    --use_sky_mask          \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_seq_hsm           \
		    --use_superglue_eval    \
		    --use_dispfeatvol_loss  \
		    --y_axis_euler_loss     \
