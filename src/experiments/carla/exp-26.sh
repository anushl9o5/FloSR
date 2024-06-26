#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
python src/train.py --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --dataset               carla \
                    --height                512 \
                    --width                 896 \
                    --exp_num               26 \
                    --exp_name              stereo_pose_estimation \
                    --exp_metainfo          siamese_9D_carla_lr_2e-6_subset_R_perturb_epipolar_loss_512x1024 \
                    --batch_size            1 \
                    --num_workers           2 \
		    --pose_repr		    use_perturb \
                    --rotation              1e+02 \
                    --learning_rate         2e-06 \
                    --model_name            siamese \
                    --num_epochs            80 \
		    --constrain_roi         \
