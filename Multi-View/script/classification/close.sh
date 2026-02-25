export CUDA_VISIBLE_DEVICES=1
export CKPT_DIR="/gemini/code/SP500/Fusion/Vison/ckpt/"
export VM_ARCH="mae_base"
export CONTEXT_LEN=336
export PERIODICITY=7
export ALIGN_CONST=0.4
export NORM_CONST=0.4
export PRED_LEN=30
python -u run.py \
    --task_name classification \
    --is_training 1 \
    --model Fusion \
    --save_dir result/stock_Close_c_336 \
    --model_id VisionTS_Close \
    --data STOCK \
    --features 'Close' \
    --target 'Risk_Label' \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128 \
    --window_size $CONTEXT_LEN \
    --vm_arch $VM_ARCH \
    --vm_ckpt $CKPT_DIR \
    --seq_len $CONTEXT_LEN \
    --periodicity $PERIODICITY \
    --pred_len $PRED_LEN \
    --norm_const $NORM_CONST \
    --align_const $ALIGN_CONST
