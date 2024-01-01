GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
CHECKPOINT_PATH=/root/autodl-tmp/hf/ckpt
VOCAB_FILE=vocab.json
MERGE_FILE=merges.txt
DATA_PATH=/root/autodl-tmp/codeparrot-full/codeparrot_content_document
GPT_ARGS="--num-layers 12
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
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
torchrun $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --attention-softmax-in-fp32 \
        --transformer-impl transformer_engine \
        $TENSORBOARD_ARGS

