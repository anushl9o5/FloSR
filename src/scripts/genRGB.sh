#!/bin/sh
python src/dataset/raw2rgb.py --frame_data_dir /nas/EOS/users/aman/data/RelativePose/frame_data \
                              --result_dir   /nas/EOS/users/aman/data/RelativePose/frames \
                              --workers 10 \
                              --start 0 \
                              --end 20000