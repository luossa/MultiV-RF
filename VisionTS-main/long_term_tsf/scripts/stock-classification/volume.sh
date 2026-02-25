export CUDA_VISIBLE_DEVICES=1
export CKPT_DIR="../ckpt/"
export VM_ARCH="mae_base"
export CONTEXT_LEN=192
export PERIODICITY=7
export ALIGN_CONST=0.4
export NORM_CONST=0.4
export PRED_LEN=30
python -u run.py \
    --task_name classification \
    --is_training 1 \
    --model VisionTS \
    --save_dir result/stock_Volume_c \
    --model_id VisionTS_Volume \
    --data STOCK \
    --features 'Volume' \
    --target 'Risk_Label' \
    --train_epochs 10 \
    --patience 10 \
    --batch_size 128 \
    --vm_arch $VM_ARCH \
    --vm_ckpt $CKPT_DIR \
    --seq_len $CONTEXT_LEN \
    --periodicity $PERIODICITY \
    --pred_len $PRED_LEN \
    --window_size $CONTEXT_LEN \
    --norm_const $NORM_CONST \
    --align_const $ALIGN_CONST