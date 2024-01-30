export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast   \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast   \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_192_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 192 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast   \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_336_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast   \
  --is_training 1 \
  --task_name long_term_forecast   \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_720_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1