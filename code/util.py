# bunch of util codes
import time
import sys
import operator
from collections import defaultdict
import tensorflow as tf
import numpy as np
import json


# taken from kelvin guu's traversing_knowledge_graphs repo, (since I really liked it).
def verboserate(iterable, time_wait=5, report=None):
    """
    Iterate verbosely.
    """
    try:
        total = len(iterable)
    except TypeError:
        total = '?'

    def default_report(steps, elapsed):
        print('{} of {} processed ({} s)'.format(steps, total, elapsed))
        sys.stdout.flush()

    if report is None:
        report = default_report

    start = time.time()
    prev = start
    for steps, val in enumerate(iterable):
        current = time.time()
        since_prev = current - prev
        elapsed = current - start
        if since_prev > time_wait:
            report(steps, elapsed)
            prev = current
        yield val

#util for sorting keys with decreasing freq of values
sort_keys = (lambda x: sorted(x.items(), key=operator.itemgetter(1), reverse=True))

#util for getting the last relevant output from output of dynamic_rnn's
def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = tf.shape(output)[2]
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def num_facts_statistics(input_file):
    """Check the distribution of number of facts"""
    bin_map = defaultdict(int)
    max_facts = -1
    with open(input_file) as f_in:
        for line in verboserate(f_in):
            q_json = json.loads(line)
            num_facts = q_json['num_facts']
            if num_facts > max_facts:
                max_facts = num_facts
            if num_facts < 10:
                bin_map['0-10'] += 1
            elif num_facts < 100:
                bin_map['10-100'] += 1
            elif num_facts < 500:
                bin_map['100-500'] += 1
            elif num_facts < 1000:
                bin_map['500-1000'] += 1
            elif num_facts < 10000:
                bin_map['1000-10000'] += 1
            elif num_facts < 20000:
                bin_map['10000-20000'] += 1
            elif num_facts < 25000:
                bin_map['20000-25000'] += 1
            elif num_facts < 30000:
                bin_map['25000-30000'] += 1
            else:
                bin_map['> 30000'] += 1
    print('Max facts {0:10d}'.format(max_facts))
    return bin_map

def read_model_predictions(precition_file, entity_vocab_file, dev_file):

    #read the answers in a list
    question_list = []
    with open(dev_file) as dev:
        for line in dev:
            line = line.strip()
            dev_q = json.loads(line)
            question = dev_q['sentence']
            question_list.append(question)

    entity_vocab = {}
    with open(entity_vocab_file) as f_in:
        entity_vocab = json.load(f_in)
    rev_entity_vocab = {}
    for k,v in entity_vocab.iteritems():
        rev_entity_vocab[v] = k

    data = np.load(precition_file)
    data = data.reshape(num_dev, -1)
    print data.shape
    seq_len = data.shape[1]
    num = data.shape[0]
    for i in range(num):
        predicted_answer = rev_entity_vocab[data[i][seq_len-2]]
        correct_answer = rev_entity_vocab[data[i][seq_len-1]]
        str = question_list[i]+' '+predicted_answer+' '+correct_answer
        if predicted_answer == correct_answer:
            print(str)

if __name__ == "__main__":
    # input_file = "/iesl/canvas/rajarshi/data/TextKBQA/dev_with_facts.json"
    # print(num_facts_statistics(input_file))
    # input_file = "/iesl/canvas/rajarshi/data/TextKBQA/train_with_facts.json"
    # print(num_facts_statistics(input_file))
    prediction_file = "/home/rajarshi/research/joint-text-and-kb-inference-semantic-parsing/out/2017.01.14-15.52.14/out_txt.0.21875"
    entity_vocab_file = "/home/rajarshi/research/joint-text-and-kb-inference-semantic-parsing/vocab/entity_vocab.json"
    dev_file = "/iesl/canvas/rajarshi/data/TextKBQA/very_small_dev_with_facts.json"
    read_model_predictions(prediction_file, entity_vocab_file, dev_file)


