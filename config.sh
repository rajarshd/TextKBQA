#!/usr/bin/env bash
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
ROOT_DIR=$(pwd)
vocab_dir="$ROOT_DIR/vocab/"
kb_file="$ROOT_DIR/kb/small_demo_kb.txt"
text_kb_file="$ROOT_DIR/text_kb/small_demo_text_kb.txt"
train_file="$ROOT_DIR/data_formatted/small_train_with_kb_and_text_facts.json"
dev_file="$ROOT_DIR/data_formatted/small_dev_with_kb_and_text_facts.json"
combine_text_kb_answer='batch_norm'
CANVAS_DIR="$ROOT_DIR/expt_outputs"
OUTPUT_DIR=$CANVAS_DIR/demo_run/${current_time}
load_model=0
model_path='path/to/trained/model' # path to trained model
load_pretrained_vectors=0
pretrained_vector_path='/path/to/pretrained/vectors'
use_kb=1
use_text=1
gpu_id=0
dev_batch_size=32
dev_eval_counter=500
save_counter=1000
batch_size=32
entity_vocab_size=1817565
relation_vocab_size=721
max_facts=5000
dev_max_facts=5000
max_text_facts=2500
dev_max_text_facts=5000
embedding_dim=50
min_facts=0
learning_rate=1e-3
grad_clip_norm=5
verbose=1
hops=1
separate_key_lstm=0
mode='train' #set this to train or test
#mode='test' #set this to train or test
create_expt_dir=1  #make it 0 if you dont want to creat an output directory and only print stuff

if [ $create_expt_dir -eq 1 ]; then
    mkdir -p $OUTPUT_DIR
else
    echo "WARNING!!! - create_expt_dir is not set. No output will be written."
fi
print_attention_weights=0