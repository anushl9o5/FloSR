#!/bin/sh
export CUDA_VISIBLE_DEVICES=6,7
python src/train.py --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
		    --load_weights_folder /nas/EOS/users/anush/logs/stereo_pose_estimation/exp-13_siamese_6D_carla_lr_0.00002_subset_radial_mask_512x1024/models/weights_0 \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --dataset               carla \
                    --height                512 \
                    --width                 1024 \
                    --exp_num               20 \
                    --exp_name              stereo_pose_estimation \
                    --exp_metainfo          siamese_6D_carla_lr_2e-5_subset_512x1024 \
                    --batch_size            1 \
                    --num_workers           2 \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            siamese \
                    --num_epochs            80 \
                    --test_res_opt_flow     0.4 \
                    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --use_radial_flow_mask  \
		    --constrain_roi
		    --use_subset
