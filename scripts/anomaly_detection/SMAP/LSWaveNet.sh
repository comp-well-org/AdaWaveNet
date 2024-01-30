export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model LSWaveNet \
  --data SMAP \
  --features M \
  --seq_len 100 \
  --pred_len 100 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 0.5 \
  --train_epochs 10\
  --lifting_levels 1\
  --lifting_kernel_size 7\
  --n_cluster 1\
  --learning_rate 0.0005\
  --batch_size 128