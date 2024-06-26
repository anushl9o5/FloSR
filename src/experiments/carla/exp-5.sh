#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
python src/train.py --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/aman/logs \
                    --dataset               carla \
                    --height                256 \
                    --width                 512 \
                    --exp_num               8 \
                    --exp_name              stereo_pose_estimation \
                    --exp_metainfo          siamese_6D_deep_carla_lr_0.0002_visual_loss \
                    --batch_size            8 \
                    --num_workers           2 \
                    --rotation              1e+02 \
                    --learning_rate         5e-04 \
                    --model_name            siamese \
                    --num_epochs            80 \
                    --use_visual_loss


