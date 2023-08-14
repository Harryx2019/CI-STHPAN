

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/StockForecasting" ]; then
    mkdir ./logs/StockForecasting
fi
seq_len=512
pred_len=1
model_name=Pretrain_graph_wo_revin_ci

# model_id_name=NASDAQ
model_id_name=NYSE

data_name=stock

random_seed=2023

k=5
python -u ../../patchtst_pretrain.py \
      --random_seed $random_seed \
      --market $model_id_name \
      --context_points $seq_len \
      --target_points $pred_len \
      --graph 1 \
      --rel_type 3 \
      --k $k \
      --revin 0 \
      --ci 0 \
      --n_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --n_epochs_pretrain 100 \
      --batch_size 1 \
      --lr 0.0001 >logs/StockForecasting/$model_name'_'$model_id_name'_'$seq_len'_dtw'$k.log