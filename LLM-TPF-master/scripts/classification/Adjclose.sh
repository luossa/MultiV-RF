export CUDA_VISIBLE_DEVICES=0
export len=192
model=TPF

python run.py \
    --features 'Adj Close' \
    --target 'Risk_Label' \
    --window_size $len \
    --window_stride 1 \
    --is_training 1 \
    --num_class 5 \
    --task_name classification \
    --model_id 'Adj Close_192_c' \
    --data STOCK \
    --seq_len $len \
    --label_len 0 \
    --pred_len 1 \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 1 \
    --c_out 1 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10 \
    --task_loss ce \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1

echo '====================================================================================================================='

