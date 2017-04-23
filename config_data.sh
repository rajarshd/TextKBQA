#!/bin/bash
#dir to store the int mappings 
ROOT_DIR=$(pwd)
vocab_dir="/home/rajarshi/research/joint-text-and-kb-inference-semantic-parsing/vocab_spades"
kb_file="/iesl/canvas/rajarshi/data/SPADES_NEW/freebase.spades.txt.new"
text_kb_file="/iesl/canvas/rajarshi/data/SPADES_NEW/text_kb.spades.txt"
train_file="/iesl/canvas/rajarshi/data/SPADES_NEW/train_with_kb_facts.json"
dev_file="/iesl/canvas/rajarshi/data/SPADES/dev_with_kb_facts.json"
test_file="/iesl/canvas/rajarshi/data/SPADES/test_with_kb_facts.json"
#because of the design, keep one of them on at a time.
make_vocab=0 #1 to create new vocabs; 0 means you want to reuse a previously created ones
extract_relevant_kb=0 #to extract part of KB which occur in train set
sort_freebase=0 #sort freebase wrt e1
make_text_kb=0 #1 to create text kb from train file
augment_kb_facts=0 #1 to augment files with kb facts
augment_text_kb_facts=0 #1 to augment files with text kb facts