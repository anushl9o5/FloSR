#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
python src/train.py --data_dir          /nas/EOS/users/aman/data/RelativePose/frames \
                    --gt_dir            /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
                    --splits_dir        src/data/split_files \
                    --log_dir           /nas/EOS/users/aman/logs \
                    --exp_num           0 \
                    --exp_name          stereo_pose_estimation \
                    --exp_metainfo      pose_cnn \
                    --batch_size        32 \
                    --num_workers       8 \
                    --rotation          1e+01 \
                    --learning_rate     5e-04 \
                    --use_quat_repr





