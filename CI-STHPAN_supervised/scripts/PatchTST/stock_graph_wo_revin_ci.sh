
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/StockForecasting" ]; then
    mkdir ./logs/StockForecasting
fi
seq_len=336
pred_len=1
model_name=PatchTST

root_path_name=../../dataset/
data_path_name=stock/2013-01-01

# model_id_name=NASDAQ
# k=20
model_id_name=NYSE
k=5

data_name=stock
random_seed=2023

for alpha in 1 2 4 6 8 10
do
    python -u ../../run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --market $model_id_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features MS \
        --target Close\
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 5 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --ci 0\
        --revin 0\
        --des 'Exp' \
        --train_epochs 100\
        --patience 10\
        --lradj 'TST'\
        --pct_start 0.2\
        --gpu 0\
        --graph 1\
        --rel_type 3\
        --k $k\
        --alpha $alpha\
        --itr 5 --batch_size 1 --learning_rate 0.0001 >logs/StockForecasting/$model_name'_'$model_id_name'_'$seq_len'_k'$k'_wo_RevIN_CI_'$alpha.log
done