#!/bin/sh
export CUDA_VISIBLE_DEVICES=2,3
python src/train.py \
	            --dataset               sintel \
	            --data_dir              /home/eos/workspace/nas.anush/SintelStereo \
                    --gt_dir                /home/eos/workspace/nas.anush/SintelStereo \
		    --splits_dir            src/data/split_files/sintel/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               98 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          sintel_9D_R6_gmflow \
                    --batch_size            6 \
                    --num_workers           12 \
		    --pose_repr		    use_9D \
                    --rotation              1e+02 \
                    --learning_rate         1e-05 \
                    --model_name            gmflow \
                    --num_epochs            120 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     1 \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_superglue_eval    \
		    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
