#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,5
python src/train.py --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --load_weights_folder /nas/EOS/users/anush/logs/stereo_pose_estimation/exp-10_siamese_6D_carla_lr_0.0002_visual_loss_512x1024/models/weights_0 \
                    --dataset               carla \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               10 \
                    --exp_name              stereo_pose_estimation \
                    --exp_metainfo          siamese_6D_carla_lr_0.0002_visual_loss_512x1024 \
                    --batch_size            1 \
                    --num_workers           2 \
                    --rotation              1e+02 \
                    --learning_rate         2e-04 \
                    --model_name            siamese \
                    --num_epochs            80 \
                    --test_res_opt_flow     0.4 \
                    --use_opt_flow_loss


