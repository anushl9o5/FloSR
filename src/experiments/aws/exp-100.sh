#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5
python src/train.py \
	            --dataset               kitti \
	            --data_dir              /home/eos/workspace/kitti \
                    --gt_dir                /home/eos/workspace/kitti \
		    --splits_dir            src/data/split_files/kitti/ \
                    --log_dir               /home/eos/worskpace/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               100 \
                    --exp_name              aws \
                    --exp_metainfo          kitti_6D_gmflow \
                    --batch_size            2 \
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
		    --load_weights_folder   /home/eos/workspace/logs/gmflow_ablations/exp-100_kitti_6D_gmflow_no_OF/models/weights_18 \
                    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \


