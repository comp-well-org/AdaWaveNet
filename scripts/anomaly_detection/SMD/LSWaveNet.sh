export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model LSWaveNet \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 100 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 1 \
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --train_epochs 10\
  --lifting_levels 1\
  --lifting_kernel_size 7\
  --n_cluster 1\
  --learning_rate 0.0005\
  --batch_size 128