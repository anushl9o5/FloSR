#!/bin/sh
export CUDA_VISIBLE_DEVICES=5,6
python src/infer.py \
	            --dataset               torc \
	            --data_dir              /home/eos/workspace/data/cvpr_torc2/torc \
                --gt_dir                /home/eos/workspace/data/RelativePose/groundtruth_json_poses \
		        --splits_dir            src/data/split_files/cvpr_videos2/torc/wk3/CAPT-353/CAPT-353.csv \
                    --log_dir               /home/eos/workspace/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               101 \
                    --exp_name              sequences \
                    --exp_metainfo          torc_6D_gmflow_CAPT-353 \
                    --batch_size            4 \
                    --num_workers           12 \
		            --pose_repr		    use_6D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            gmflow \
                    --num_epochs            160 \
                    --algolux_test_res_opt_flow     0.5 \
		            --resize_orig_img       \
                    --save_imgs \
		    --load_weights_folder   /home/eos/workspace/logs/hsm_debug/exp-101_torc_6D_gmflow/models/weights_21
