#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5
python src/train.py \
	            --dataset               carla \
	            --data_dir              /home/eos/workspace/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /home/eos/workspace/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
		    --splits_dir            src/data/split_files/carla/ \
                    --log_dir               /home/eos/workspace/logs \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               99 \
                    --exp_name              aws_train \
                    --exp_metainfo          carla_no_init \
                    --batch_size            3 \
                    --num_workers           12 \
		    --pose_repr		    use_9D \
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
                    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
            --ablate_init \
            --load_weights_folder /home/eos/workspace/logs/aws_train/exp-99_carla_no_init/models/weights_0/ \

