#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5
python src/infer.py \
	            --dataset               argo2 \
	            --data_dir              /nas/EOS/users/anush/argo/ \
                    --gt_dir                /nas/EOS/users/anush/argo/ \
		    --splits_dir            src/data/split_files/argoverse2/ \
                    --log_dir               /nas/EOS/users/anush/logs \
                    --height                768 \
                    --width                 1024 \
                    --exp_num               89 \
                    --exp_name              hsm_debug \
                    --exp_metainfo          argo2_9D_seq_R6_costvol_OF \
                    --batch_size            16 \
                    --num_workers           12 \
		    --pose_repr		    use_9D \
                    --rotation              1e+02 \
                    --learning_rate         2e-05 \
                    --model_name            hsm \
                    --num_epochs            80 \
                    --algolux_test_res_opt_flow     0.5 \
		    --constrain_roi         \
		    --epoch_visual_loss     0 \
		    --use_sky_mask          \
		    --use_radial_flow_mask  \
		    --resize_orig_img       \
		    --use_seq_hsm           \
		    --use_superglue_eval    \
		    --use_dispfeatvol_loss  \
		    --use_opt_flow_loss     \
		    --multi_gpu_opt_flow    \
		    --load_weights_folder  /home/eos/workspace/nas.anush/logs/hsm_debug/exp-89_argo2_9D_seq_R6_costvol_OF/models/weights_79 \

