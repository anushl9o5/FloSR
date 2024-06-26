#!/bin/sh
export CUDA_VISIBLE_DEVICES=5,4
python src/infer.py \
	            --dataset               kitti \
	            --data_dir              /home/eos/workspace/nas.anush/kitti \
                    --gt_dir                /home/eos/workspace/nas.anush/kitti \
		    --splits_dir            src/data/split_files/kitti/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                512 \
                    --width                 1344 \
                    --exp_num               95 \
                    --exp_name              hsm_infer \
                    --exp_metainfo          kitti_9D_R6_costvol_of \
                    --batch_size            1 \
                    --num_workers           12 \
		    --pose_repr		    use_9D \
                    --rotation              1e+02 \
                    --learning_rate         1e-05 \
                    --model_name            hsm \
                    --num_epochs            120 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     1 \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_seq_hsm           \
		    --use_superglue_eval    \
		    --use_dispfeatvol_loss  \
		    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --load_weights_folder   /home/eos/workspace/nas.anush/logs/hsm_debug/exp-95_kitti_9D_R6_costvol_of/models/weights_23 \

