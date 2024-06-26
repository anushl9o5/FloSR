#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
python src/train.py --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --dataset               carla \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               17 \
                    --exp_name              stereo_pose_estimation \
                    --exp_metainfo          siamese_6D_carla_lr_2e-5_subset_gt_only_512x1024 \
                    --batch_size            1 \
                    --num_workers           2 \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            siamese \
                    --num_epochs            80 \
		    --use_subset
