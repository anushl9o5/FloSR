#!/bin/sh
export CUDA_VISIBLE_DEVICES=1
python src/train.py --data_dir          /nas/EOS/users/aman/data/RelativePose/frames \
                    --gt_dir            /nas/EOS/users/aman/data/RelativePose/groundtruth_json_poses \
                    --splits_dir        test_split \
                    --log_dir           /nas/EOS/users/anush/logs \
                    --exp_num           0 \
                    --exp_name          stereo_pose_estimation \
                    --exp_metainfo      pose_cnn_verfit_test1 \
                    --batch_size        1 \
                    --num_workers       0 \
                    --rotation          1e+01 \
                    --learning_rate     5e-04 \
                    --num_epochs        1000 \





