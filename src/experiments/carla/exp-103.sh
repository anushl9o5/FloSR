#!/bin/sh
export CUDA_VISIBLE_DEVICES=2,3
python src/train.py \
	            --dataset               flyingthings \
	            --data_dir              /home/eos/workspace/nas.anush/flyingthings/cleanpass/frames_cleanpass \
                    --gt_dir                /home/eos/workspace/nas.anush/flyingthings/camera_data \
		    --splits_dir            src/data/split_files/flyingthings/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               103 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          flyingthings_6D_gmflow \
                    --batch_size            6 \
                    --num_workers           12 \
		    --pose_repr		    use_6D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            gmflow \
                    --num_epochs            160 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     1 \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_superglue_eval    \
		    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --superglue_weights     indoor \
