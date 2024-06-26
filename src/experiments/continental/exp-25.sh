#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,5
python src/inference.py \
	            --data_dir              /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
		    --load_weights_folder      /nas/EOS/users/anush/logs/stereo_pose_estimation/exp-25_sterepose_9D_lr_2e-6_perturb_0.2_optflow_544x960/models/weights_5 \
                    --splits_dir            src/data/split_files/carla \
                    --log_dir               /nas/EOS/users/anush/logs \
		    --pose_repr		    use_perturb \
                    --dataset               continental \
                    --height                1440 \
                    --width                 2560 \
                    --exp_num               25 \
                    --exp_name              superglue_eval \
                    --exp_metainfo          conti_superglue_perturb_0.2_eval \
                    --batch_size            1 \
                    --num_workers           2 \
                    --rotation              1e+02 \
                    --learning_rate         2e-04 \
                    --model_name            superglue \
                    --num_epochs            80 \


