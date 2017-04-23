import json
import util
from collections import defaultdict


def get_fb_stats(freebase_data_file):
    with open(freebase_data_file) as fb:
        fact_counter = 0
        relation_set = set()
        entity_set = set()
        for line in fb:
            line = line.strip()
            line = line[1:-1]
            e1, r1, r2, e2 = [a.strip('"') for a in [x.strip() for x in line.split(',')]]
            r = r1 + '_' + r2
            fact_counter += 1
            relation_set.add(r)
            entity_set.add(e1)
            entity_set.add(e2)

    print("Total num of facts {}".format(fact_counter))
    print("Num unique entities {}".format(len(entity_set)))
    print("Num unique relations {}".format(len(relation_set)))


def get_questions_stats(train_data_file, dev_data_file):
    print('1. Getting the number of blanks')

    blank_str = '_blank_'
    num_blanks_map = defaultdict(int)
    word_freq_train = defaultdict(int)
    with open(train_data_file) as train_file:
        for counter, line in enumerate(util.verboserate(train_file)):
            line = line.strip()
            q_json = json.loads(line)
            q = q_json['sentence']
            count = q.count(blank_str)
            num_blanks_map[count] += 1
            words = q.split(' ')
            for word in words:
                word = word.strip()
                word_freq_train[word] += 1
            a_list = q_json['answerSubset']
            for a in a_list:
                word_freq_train[a] = word_freq_train[word] + 1

    print(num_blanks_map)

    print '2. Number of word types in the train set {}'.format(len(word_freq_train))

    print '3. Checking overlap with the dev answers'
    dev_answers_present = set()
    dev_answers_oov = set()
    dev_answers = set()
    with open(dev_data_file) as dev_file:
        for line in dev_file:
            line = line.strip()
            dev_json = json.loads(line)
            a_list = dev_json['answerSubset']
            for a in a_list:
                if a in word_freq_train:
                    dev_answers_present.add(a)
                else:
                    dev_answers_oov.add(a)
                dev_answers.add(a)

    print 'Number of unique dev answer strings {}'.format(len(dev_answers))

    print 'Number of oov answer strings in dev set {}'.format(len(dev_answers_oov))

    print 'Number of dev answer strings which have atleast 1 occurrences in train set {}'.format(
        len(dev_answers_present))


freebase_data_file = "/home/rajarshi/research/graph-parser/data/spades/freebase.spades.txt"
train_data_file = "/home/rajarshi/research/graph-parser/data/spades/train.json"
dev_data_file = "/home/rajarshi/research/graph-parser/data/spades/dev.json"
# get_fb_stats()
get_questions_stats(train_data_file, dev_data_file)
