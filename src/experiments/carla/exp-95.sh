#!/bin/sh
export CUDA_VISIBLE_DEVICES=5,4
python src/train.py \
	            --dataset               kitti \
	            --data_dir              /home/eos/workspace/nas.anush/kitti \
                    --gt_dir                /home/eos/workspace/nas.anush/kitti \
		    --splits_dir            src/data/split_files/kitti/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                512 \
                    --width                 1344 \
                    --exp_num               95 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          kitti_9D_R6_costvol_of \
                    --batch_size            16 \
                    --num_workers           12 \
		    --pose_repr		    use_9D \
                    --rotation              1e+02 \
                    --learning_rate         1e-05 \
                    --model_name            hsm \
                    --num_epochs            120 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     1 \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_seq_hsm           \
		    --use_superglue_eval    \
		    --use_dispfeatvol_loss  \
		    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
