
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/StockForecasting" ]; then
    mkdir ./logs/StockForecasting
fi

random_seed=2023
model_name=Transformer

pred_len=1

root_path_name=../dataset/
data_path_name=stock/2013-01-01
model_id_name=NASDAQ
# model_id_name=NYSE


for seq_len in 8 16 32 64 128
do
    for alpha in 1 2 4 6 8 10
    do
    python -u ../run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --market $model_id_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data stock \
        --features MS \
        --target Close\
        --freq d \
        --seq_len $seq_len \
        --pred_len 1 \
        --patience 10\
        --factor 3 \
        --enc_in 5 \
        --dec_in 5 \
        --c_out 5 \
        --batch_size 1 \
        --alpha $alpha\
        --des 'Exp' \
        --itr 5 \
        --gpu 0\
        --train_epochs 100 >logs/StockForecasting/$model_name'_'$model_id_name'_'$seq_len'_a='$alpha.log
    done
done