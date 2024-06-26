#!/bin/sh
export CUDA_VISIBLE_DEVICES=2
python src/train.py --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/aman/logs \
                    --load_weights_folder   /nas/EOS/users/aman/logs/stereo_pose_estimation/exp-6_siamese_6D_carla_lr_0.0002_2/models/weights_79 \
                    --dataset               carla \
                    --height                256 \
                    --width                 512 \
                    --exp_num               9 \
                    --exp_name              stereo_pose_estimation \
                    --exp_metainfo          siamese_6D_carla_lr_0.0002_visual_loss_resized_img \
                    --batch_size            8 \
                    --num_workers           2 \
                    --rotation              1e+02 \
                    --learning_rate         2e-04 \
                    --model_name            siamese \
                    --num_epochs            80 \
                    --use_visual_loss


