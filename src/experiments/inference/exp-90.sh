#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5
python src/infer.py \
	            --dataset               dstereo \
	            --data_dir              /nas/EOS-DB/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06 \
                    --gt_dir                /nas/EOS-DB/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/cam_poses \
		    --splits_dir            src/data/split_files/dstereo/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                384 \
                    --width                 832 \
                    --exp_num               90 \
                    --exp_name              hsm_infer \
                    --exp_metainfo          dstereo_9D_R6_costvol_of \
                    --batch_size            1 \
                    --num_workers           12 \
		    --pose_repr		    use_9D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            hsm \
                    --num_epochs            160 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     0 \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_seq_hsm           \
		    --use_superglue_eval    \
		    --y_axis_euler_loss     \
		    --use_dispfeatvol_loss  \
		    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --load_weights_folder   /home/eos/workspace/nas.anush/logs/hsm_debug/exp-90_dstereo_9D_R6_costvol_of/models/weights_159/ \

