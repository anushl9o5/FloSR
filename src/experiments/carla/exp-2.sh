#!/bin/sh
#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
python src/train.py --data_dir          /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir            /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
                    --splits_dir        src/data/split_files/carla \
                    --log_dir           /nas/EOS/users/aman/logs \
                    --dataset           carla \
                    --height            256 \
                    --width             512 \
                    --exp_num           5 \
                    --exp_name          stereo_pose_estimation \
                    --exp_metainfo      siamese_6D_carla_lr_0.0004 \
                    --batch_size        16 \
                    --num_workers       8 \
                    --rotation          1e+02 \
                    --learning_rate     4e-04 \
                    --model_name        siamese \
                    --num_epochs        60

