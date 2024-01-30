export CUDA_VISIBLE_DEVICES=0

model_name=LSWaveNet

python -u run.py\
   --task_name long_term_forecast   \
   --is_training 1   \
   --root_path ./dataset/illness/ \
   --data_path national_illness.csv \
   --model_id ili_36_24 \
   --model $model_name \
   --data custom \
   --features M \
   --seq_len 36 \
   --label_len 18 \
   --pred_len 36 \
   --e_layers 3   \
   --d_layers 1   \
   --factor 3   \
   --enc_in 7   \
   --dec_in 7   \
   --c_out 7   \
   --des 'Exp'   \
   --d_model 512  \
   --d_ff 512  \
   --itr 1   \
   --lifting_levels 4  \
   --lifting_kernel_size 7  \
   --n_cluster 1  \
   --learning_rate 0.0005  \
   --batch_size 32 \
   --adjust_lr True


python -u run.py\
   --task_name long_term_forecast\
   --is_training 1   \
   --root_path ./dataset/exchange_rate/   \
   --data_path exchange_rate.csv   \
   --model_id exchange_192_192   \
   --model $model_name   \
   --data custom   \
   --features M   \
   --seq_len 192   \
   --label_len 48   \
   --pred_len 192   \
   --e_layers 3   \
   --d_layers 1   \
   --factor 3   \
   --enc_in 8   \
   --dec_in 8   \
   --c_out 8   \
   --des 'Exp'   \
   --d_model 512  \
   --d_ff 512  \
   --itr 1   \
   --lifting_levels 5  \
   --lifting_kernel_size 7  \
   --n_cluster 1  \
   --learning_rate 0.0005  \
   --batch_size 16 \
   --adjust_lr True


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/   \
  --data_path exchange_rate.csv   \
  --model_id exchange_336_336   \
  --model $model_name   \
  --data custom   \
  --features M   \
  --seq_len 336   \
  --label_len 48   \
  --pred_len 336   \
  --e_layers 3   \
  --d_layers 1   \
  --factor 3   \
  --enc_in 8   \
  --dec_in 8   \
  --c_out 8   \
  --des 'Exp'   \
  --d_model 512  \
  --d_ff 512  \
  --itr 1   \
  --lifting_levels 4  \
  --lifting_kernel_size 7  \
  --n_cluster 1  \
  --learning_rate 0.0005  \
  --batch_size 16 \
  --adjust_lr True


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/   \
  --data_path exchange_rate.csv   \
  --model_id exchange_720_720   \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3   \
  --d_layers 1   \
  --factor 3   \
  --enc_in 8   \
  --dec_in 8   \
  --c_out 8   \
  --des 'Exp'   \
  --d_model 512  \
  --d_ff 512  \
  --itr 1   \
  --lifting_levels 1  \
  --lifting_kernel_size 7  \
  --n_cluster 1  \
  --learning_rate 0.0005  \
  --batch_size 32 \
  --adjust_lr True