#!/usr/bin/env bash

source $1
mkdir -p $vocab_dir

#system variables
#PYTHON_EXEC="~/anaconda2/bin/python"
#data processing..

python_path="/usr/bin/python"

cmd="$python_path -u $ROOT_DIR/code/train.py --train_file $train_file --dev_file $dev_file \
--kb_file $kb_file --text_kb_file $text_kb_file --vocab_dir $vocab_dir --max_facts $max_facts --min_facts $min_facts \
--max_text_facts $max_text_facts --dev_max_facts $dev_max_facts --dev_max_text_facts $dev_max_text_facts
--entity_vocab_size $entity_vocab_size --relation_vocab_size $relation_vocab_size \
--learning_rate $learning_rate --verbose $verbose --embedding_dim $embedding_dim --grad_clip_norm $grad_clip_norm \
--hops $hops --dev_batch_size $dev_batch_size --batch_size $batch_size --output_dir $OUTPUT_DIR \
--load_model $load_model --model_path $model_path --load_pretrained_vectors $load_pretrained_vectors \
--pretrained_vector_path $pretrained_vector_path --save_counter $save_counter --dev_eval_counter $dev_eval_counter
--use_kb $use_kb --use_text $use_text --print_attention_weights $print_attention_weights --mode $mode \
--combine_text_kb_answer $combine_text_kb_answer --separate_key_lstm $separate_key_lstm"


if [ $create_expt_dir -eq 1 ]; then
    set > $OUTPUT_DIR/config.txt
    echo "Executing $cmd" | tee $OUTPUT_DIR/log.txt.$current_time
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd | tee -a $OUTPUT_DIR/log.txt.$current_time
    #print the configs
else
    echo "Executing $cmd"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd
fi

if [ $print_attention_weights -eq 1 ]; then
    cmd="$python_path -u $ROOT_DIR/code/qual_eval.py --use_kb $use_kb --use_text $use_text --kb_file $kb_file
    --text_kb_file $text_kb_file --attn_file $OUTPUT_DIR/attn_wts.npy --input_test_file $dev_file
    --answer_file $OUTPUT_DIR/out.txt --output_dir $OUTPUT_DIR"
 echo "Executing $cmd"
 $cmd
fi

