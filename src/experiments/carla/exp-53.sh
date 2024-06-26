#!/bin/sh
export CUDA_VISIBLE_DEVICES=6,7
python src/train.py --data_dir              /nas/EOS/users/aman/data/RelativePose/frames \
                    --gt_dir                /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --load_weights_folder   /nas/EOS/users/anush/logs/hsm_debug/exp-53_sterepose_6D_lr_2e-5_optflow_576x768/models/weights_1/ \
		    --splits_dir            src/data/split_files/algolux \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                576 \
                    --width                 768 \
                    --exp_num               53 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          sterepose_6D_lr_2e-5_optflow_576x768 \
                    --batch_size            8 \
                    --num_workers           4 \
		    --pose_repr		    use_6D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --test_res_opt_flow     0.3334 \
		    --use_opt_flow_loss     \
		    --use_radial_flow_mask  \
		    --constrain_roi         \
		    --resize_orig_img       \
		    --y_axis_euler_loss     \
		    --epoch_visual_loss     0 \
