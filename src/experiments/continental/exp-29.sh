#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,5
python eval.py \
	            --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/anush/logs \
		    --pose_repr		    superglue \
                    --dataset               continental \
                    --height                1440 \
                    --width                 2560 \
                    --exp_num               29 \
                    --exp_name              superglue_eval \
                    --exp_metainfo          conti_pitch_eval \
                    --batch_size            1 \
                    --num_workers           2 \
                    --rotation              1e+02 \
                    --learning_rate         2e-04 \
                    --model_name            superglue \
                    --num_epochs            80 \


