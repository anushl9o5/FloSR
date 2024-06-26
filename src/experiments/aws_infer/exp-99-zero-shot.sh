#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5
python src/infer.py \
	            --dataset               kitti \
	            --data_dir              /home/eos/workspace/data/kitti \
                    --gt_dir                /home/eos/workspace/data/kitti \
		    --splits_dir            src/data/split_files/kitti/ \
                    --log_dir               /home/eos/workspace/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               99 \
                    --exp_name              aws_infer \
                    --exp_metainfo          carla_zero_shot_kitti \
                    --batch_size            1 \
                    --num_workers           12 \
		    --pose_repr		    use_6D \
                    --rotation              1e+02 \
                    --learning_rate         1e-05 \
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
            --load_weights_folder /home/eos/workspace/logs/hsm_debug/exp-99_carla_6D_gmflow/models/weights_94/ \
