#!/bin/bash
source $1

mkdir -p $vocab_dir

#system variables
PYTHON_EXEC="/share/apps/python/bin/python"
#data processing..

cmd="python $ROOT_DIR/code/data_utils.py --make_vocab $make_vocab --vocab_dir $vocab_dir --kb_file $kb_file\
 --text_kb_file $text_kb_file --train_file $train_file --dev_file $dev_file --test_file $test_file \
 --extract_relevant_kb $extract_relevant_kb --make_text_kb $make_text_kb \
 --augment_text_kb_facts $augment_text_kb_facts --augment_kb_facts $augment_kb_facts --sort_freebase $sort_freebase"

echo "Executing $cmd"
$cmd