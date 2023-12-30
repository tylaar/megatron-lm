CHECKPOINT_PATH=/opt/hf/ckpt
VOCAB_FILE=vocab.json
MERGE_FILE=merges.txt
CUDA_DEVICE_MAX_CONNECTIONS=1
DATA_PATH=/home/oscar/hack/Megatron-LM/codeparrot_content_document
GPT_ARGS="--num-layers 4
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 150000
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999
--bf16
--log-interval 10
--save-interval 2000
--eval-interval 200
--eval-iters 10
"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
torchrun --standalone --nnodes=1 \
        pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
	--fp8-format hybrid \
        --attention-softmax-in-fp32 \
	--transformer-impl transformer_engine \
        $TENSORBOARD_ARGS


