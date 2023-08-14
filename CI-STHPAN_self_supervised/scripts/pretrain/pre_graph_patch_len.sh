

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/StockForecasting" ]; then
    mkdir ./logs/StockForecasting
fi
seq_len=512
pred_len=1
model_name=Pretrain_graph_patch_len

model_id_name=NASDAQ
# model_id_name=NYSE

data_name=stock

random_seed=2023

for k in 20
do
for patch_len in 12 16 24 32 40
do
python -u ../../patchtst_pretrain.py \
      --random_seed $random_seed \
      --market $model_id_name \
      --context_points $seq_len \
      --target_points $pred_len \
      --patch_len $patch_len \
      --stride $patch_len \
      --graph 1 \
      --rel_type 3 \
      --k $k \
      --n_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --head_dropout 0 \
      --n_epochs_pretrain 100 \
      --batch_size 1 \
      --lr 0.0001 >logs/StockForecasting/$model_name'_'$model_id_name'_'$seq_len'_k='$k'_patch_len='$patch_len.log
done
done