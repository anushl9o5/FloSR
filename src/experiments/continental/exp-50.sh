#!/bin/sh
export CUDA_VISIBLE_DEVICES=2
python src/inference.py \
		    --data_dir              /nas/EOS/users/aman/data/RelativePose/frames \
                    --gt_dir                /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
		    --load_weights_folder   /nas/EOS/users/anush/logs/hsm_debug/exp-50_sterepose_9D_lr_2e-5_perturb_0.2_optflow_576x768/models/weights_2/ \
		    --splits_dir            src/data/split_files/algolux \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                1440 \
                    --width                 2560 \
                    --exp_num               50 \
                    --exp_name              superglue_eval \
                    --exp_metainfo          conti_superglue_hsm_perturb_0.2_eval \
                    --batch_size            1 \
                    --num_workers           2 \
		    --pose_repr		    use_perturb \
                    --model_name            hsm \
