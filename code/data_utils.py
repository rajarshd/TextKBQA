import json, os
from collections import defaultdict
import operator
import util
import argparse
from tqdm import tqdm
import numpy as np


class KB(object):
    """Class for freebase"""
    def __init__(self, input_file, create_vocab=False, vocab_dir=''):
        self.input_file = input_file
        self.create_vocab = create_vocab
        self.vocab_dir = vocab_dir
        self.entity_vocab = None
        self.relation_vocab = None
        if self.create_vocab:
            self.entity_vocab, self.relation_vocab, self.facts = self.build_int_map()
        else:
            self.facts, self.facts_list = self.read_kb_facts()
            self.entity_vocab = json.load(open(vocab_dir+'/entity_vocab.json'))
            self.relation_vocab = json.load(open(vocab_dir + '/relation_vocab.json'))

        self.num_entities = len(self.entity_vocab) if self.entity_vocab is not None else None
        self.num_relations = len(self.relation_vocab) if self.relation_vocab is not None else None

    def read_kb_facts(self):
        facts = []
        facts_list = defaultdict(list)
        print('Reading kb file at {}'.format(self.input_file))
        with open(self.input_file) as fb:
            for counter, line in tqdm(enumerate(fb)):
                line = line.strip()
                line = line[1:-1]
                e1, r1, r2, e2 = [a.strip('\'') for a in [x.strip() for x in line.split(',')]]
                r = r1 + '_' + r2
                facts.append({'e1': e1, 'r': r, 'e2': e2})
                facts_list[e1].append(counter)  # just store the fact counter instead of the fact
        return facts, facts_list

    def build_int_map(self):
        entity_count_map = defaultdict(int)
        relation_count_map = defaultdict(int)
        facts = []
        facts_list = defaultdict(list)
        with open(self.input_file) as fb:
            for line in tqdm(fb):
                line = line.strip()
                line = line[1:-1]
                e1, r1, r2, e2 = [a.strip('\'') for a in [x.strip() for x in line.split(',')]]
                r = r1 + '_' + r2
                entity_count_map[e1] += 1
                entity_count_map[e2] += 1
                relation_count_map[r] += 1
                facts.append({'e1': e1, 'r': r, 'e2': e2})

        sort_key_by_freq = lambda x: sorted(x.items(), key=operator.itemgetter(1), reverse=True)
        entity_vocab = {k: counter + 2 for counter, (k, _) in enumerate(sort_key_by_freq(entity_count_map))}
        relation_vocab = {k: counter + 2 for counter, (k, _) in enumerate(sort_key_by_freq(relation_count_map))}
        entity_vocab['PAD'] = 0
        entity_vocab['UNK'] = 1
        relation_vocab['PAD'] = 0
        relation_vocab['UNK'] = 1
        relation_vocab['DUMMY_MEM'] = len(relation_vocab)
        #the dummy_mem key to entity_vocab is added at the end of augmenting
        # the entity_vocab with words from questions. c.f. augment_to_entity_vocab method

        return entity_vocab, relation_vocab, facts

    def save_vocab(self,**kwargs):
        assert len(self.vocab_dir) != 0
        with open(self.vocab_dir + '/entity_vocab.json', 'w') as ent_out, \
                open(self.vocab_dir + '/relation_vocab.json', 'w') as rel_out:
            json.dump(self.entity_vocab, ent_out)
            json.dump(self.relation_vocab, rel_out)

        with open(self.vocab_dir+'/stats.json','w') as f_out:
            f_out.write("Num entities {}\n".format(self.num_entities))
            f_out.write("Num relations {}\n".format(self.num_relations))
            if 'num_words' in kwargs:
                f_out.write("Num words {}\n".format(kwargs['num_words']))

class TextKb(object):
    """Class for kb formed from training data"""

    def __init__(self, input_file, create_vocab=False, vocab_dir=''):

        self.facts_list, self.entity_facts_index_map, self.max_key_length = self.parse_kb(input_file)
        self.entity_vocab = json.load(open(vocab_dir + '/entity_vocab.json'))
        self.relation_vocab = json.load(open(vocab_dir + '/relation_vocab.json'))
    def parse_kb(self, input_file):
        kb_facts = []
        entity_facts_index_map = {} #map from entity to tuple of (start_index, num_facts)
        prev_entity = None
        start_index = 0
        num_facts = 0
        max_key_length = -1
        print('Reading the text kb file...')
        with open(input_file) as fin:
            for counter, line in tqdm(enumerate(fin)):
                kb_instance = json.loads(line)
                kb_facts.append(kb_instance)
                entity = kb_instance['entity']
                key_length = int(kb_instance['key_length'])
                if key_length > max_key_length:
                    max_key_length = key_length
                if prev_entity != entity:
                    if prev_entity is not None:
                        entity_facts_index_map[prev_entity] = (start_index, num_facts)
                    start_index = counter
                    prev_entity = entity
                    num_facts = 0
                num_facts+=1
        return kb_facts, entity_facts_index_map, max_key_length


class QuestionAnswer(object):
    """Class for parsing a single question answer pair"""

    def __init__(self, json_string):
        self.json_str = json_string
        #indices are positions of entities in question str.
        self.entities = []
        self.parsed_question = self.parse_question_str(self.json_str)

    def parse_question_str(self, json_str):
        q_json = json.loads(json_str)
        question = q_json['sentence']
        answers = q_json['answerSubset']
        entities = []
        indices = []
        ret = {}
        ret['question'] = question
        ret['answers'] = answers
        for entity in q_json['entities']:
            entities.append(entity['entity'])
            self.entities.append(entity['entity'])
            indices.append(entity['index'])
        ret['entities'] = entities
        ret['indices'] = indices
        #get the memory slots
        if 'start_indices' in q_json:
            ret['start_indices'] = q_json['start_indices']
            ret['fact_lengths'] = q_json['lengths']
            ret['num_facts'] = q_json['num_facts']
            if 'text_kb_num_facts' in q_json:
                ret['text_kb_num_facts'] = q_json['text_kb_num_facts']
                ret['text_kb_start_indices'] = q_json['text_kb_start_indices']
                ret['text_kb_lengths'] = q_json['text_kb_lengths']
            if 'black_lists' in q_json:
                ret['blacklists'] = q_json['black_lists']

        return ret

    def get_supporting_KB_facts(self, KB):
        """get the supporting KB facts for this QA pair. Should be just called once."""
        # look at the entities in the question. \
        # Retrieve all facts from KB which have\
        #  the entities in the given question

        start_indices = []
        lengths = []
        for entity in self.entities:
            if entity in KB.facts_list:
                # facts.update(set(KB.facts_list[entity]))
                # KB.facts_list[entity] is a contiguous list of numbers since the KB is sorted wrt e1, hence I am storing\
                #  the start index and number
                start_index = KB.facts_list[entity][0]
                length = len(KB.facts_list[entity])
                start_indices.append(start_index)
                lengths.append(length)
        return start_indices, lengths

    def get_supporting_text_kb_facts(self, text_kb):
        """get the supporting text KB facts for this QA pair. Should be just called once."""
        start_indices = []
        fact_lengths = []
        for entity in self.parsed_question['entities']:
            if entity in text_kb.entity_facts_index_map:
                start_index, num_facts = text_kb.entity_facts_index_map[entity]
                start_indices.append(start_index)
                fact_lengths.append(num_facts)
        return start_indices, fact_lengths



class Text(object):
    """Class for each textual questions file"""

    def __init__(self, input_file, **kwargs):
        self.input_file = input_file
        self.kb = kwargs['kb'] if 'kb' in kwargs else None #the kb object its entities are tied to
        self.max_kb_facts_allowed = kwargs['max_num_facts'] if 'max_num_facts' in kwargs \
            else float('inf')
        self.min_kb_facts_allowed = kwargs['min_num_facts'] if 'min_num_facts' in kwargs \
            else 0
        self.max_text_kb_facts_allowed = kwargs['max_num_text_facts'] if 'max_num_text_facts' in kwargs \
            else float('inf')
        self.min_text_kb_facts_allowed = kwargs['min_num_text_facts'] if 'min_num_text_facts' in kwargs \
            else 0
        self.max_q_length, self.max_num_kb_facts, self.max_num_text_kb_facts, self.question_list, \
        self.num_entities, self.entity_set, self.answer_entities = self.read_and_parse()

    def read_and_parse(self):
        max_length = -1
        max_num_kb_facts = -1
        max_num_text_kb_facts = -1
        question_list = []
        set_entities = set()
        answer_entities = []
        print('Reading questions file at {}'.format(self.input_file))
        with open(self.input_file) as f_in:
            for counter, line in tqdm(enumerate(f_in)):
                line = line.strip()
                qa = QuestionAnswer(line) #parse each line
                question_str = qa.parsed_question['question']
                length_q = len(question_str.split(' '))
                max_length = max(max_length, length_q)
                num_kb_facts = qa.parsed_question['num_facts'] if 'num_facts' in qa.parsed_question else 0
                num_text_kb_facts = qa.parsed_question['text_kb_num_facts'] if 'text_kb_num_facts' in qa.parsed_question else 0
                if num_kb_facts > self.max_kb_facts_allowed:
                    num_kb_facts = self.max_kb_facts_allowed
                elif num_kb_facts < self.min_kb_facts_allowed:
                    continue
                if num_text_kb_facts > self.max_text_kb_facts_allowed:
                    num_text_kb_facts = self.max_text_kb_facts_allowed
                elif num_text_kb_facts < self.min_text_kb_facts_allowed:
                    continue

                max_num_kb_facts = max(num_kb_facts, max_num_kb_facts)
                max_num_text_kb_facts = max(num_text_kb_facts, max_num_text_kb_facts)

                entities = qa.parsed_question['entities']
                for entity in entities:
                    set_entities.add(entity)
                set_entities.add(qa.parsed_question['answers'][0])
                answer_entities.append(qa.parsed_question['answers'][0])
                question_list.append(qa)
        return max_length,max_num_kb_facts, max_num_text_kb_facts, question_list, len(set_entities), set_entities, answer_entities

    def augment_to_entity_vocab(self):
        print("Augmenting words into entity vocab")
        entity_vocab = self.kb.entity_vocab
        assert entity_vocab is not None
        count_map = defaultdict(int)
        with open(self.input_file) as f_in:
            for line in tqdm(f_in):
                line = line.strip()
                qa = QuestionAnswer(line) #parse each line
                question_str = qa.parsed_question['question']
                question_entities = qa.parsed_question['entities']
                question_indices = qa.parsed_question['indices']
                words = question_str.split(' ')
                count = 0
                for index, word in enumerate(words):
                    if count >= len(question_indices) or\
                                    index != question_indices[count]:
                        count_map[word] += 1
                    else:
                        count_map[question_entities[count]] += 1
                        count += 1
        sorted_k_v_list = util.sort_keys(count_map)
        self.num_words = len(sorted_k_v_list)
        for k,_ in sorted_k_v_list:
            if k not in entity_vocab: #question entities might already be present in entity_vocab
                entity_vocab[k] = len(entity_vocab)
        entity_vocab['DUMMY_MEM'] = len(entity_vocab)


########### stuff which would be called once #######################

def augment_qa_with_kb_facts(file_name, out_file_name, kb):
    out = open(out_file_name, 'w')
    with open(file_name) as input_file:
        for counter, line in tqdm(enumerate(input_file)):
            line = line.strip()
            qa = QuestionAnswer(line)
            start_indices, lengths = qa.get_supporting_KB_facts(kb)
            q_json = json.loads(line)
            q_json['start_indices'] = start_indices
            q_json['lengths'] = lengths
            num_facts = 0
            for length in lengths:
                num_facts += length
            q_json['num_facts'] = num_facts
            json_str = json.dumps(q_json)
            out.write(json_str + '\n')

def augment_qa_with_text_kb_facts(file_name, out_file_name, text_kb, is_train_file=False):
    out = open(out_file_name, 'w')
    with open(file_name) as input_file:
        for counter, line in tqdm(enumerate(input_file)):
            line = line.strip()
            qa = QuestionAnswer(line)
            start_indices, lengths = qa.get_supporting_text_kb_facts(text_kb)
            q_json = json.loads(line)
            q_json['text_kb_start_indices'] = start_indices
            q_json['text_kb_lengths'] = lengths
            num_facts = 0
            for length in lengths:
                num_facts += length
            q_json['text_kb_num_facts'] = num_facts

            # # since the textkb is made out of sentences in train set
            # # hence removing the memory slots which are formed of this particular example.
            # if is_train_file:
            #     q_word_list = qa.parsed_question['question'].split(' ')
            #     answer_entity = qa.parsed_question['answers'][0]
            #     #replace blank with answer entity:
            #     for word_counter, word in enumerate(q_word_list):
            #         if word == '_blank_':
            #             q_word_list[word_counter] = answer_entity
            #             break
            #     question_entities = qa.parsed_question['entities']
            #     question_indices = qa.parsed_question['indices']
            #     #replace words with entities
            #     for counter, index in enumerate(question_indices):
            #         q_word_list[index] = question_entities[counter]
            #     text_kb_fact_list = text_kb.facts_list
            #     black_lists = [] #this will be a list of lists
            #     for counter, start_index in enumerate(start_indices):
            #         each_entity_black_list = []
            #         fact_length = lengths[counter]
            #         for mem_counter, mem_index in enumerate(xrange(start_index, start_index+fact_length)):
            #             mem = text_kb_fact_list[mem_index]
            #             key = mem['key']
            #             val_entity = mem['value']
            #             for word_counter, word in enumerate(key):
            #                 if word == '_blank_':
            #                     key[word_counter] = val_entity
            #             #check if key and q_word_list are equal; if they are black list that memory
            #             if key == q_word_list:
            #                 each_entity_black_list.append(mem_counter)
            #         black_lists.append(each_entity_black_list)
            #     q_json['black_lists'] = black_lists
            json_str = json.dumps(q_json)
            out.write(json_str + '\n')


def break_input_file(input_file, start, end):
    # only select inputs which have number of facts between start and end
    out_file = input_file + '.{}.{}'.format(start, end)
    f_out = open(out_file, 'w')
    with open(input_file) as f_in:
        for line in tqdm(f_in):
            q_json = json.loads(line)
            num_facts = q_json['num_facts']
            if start <= num_facts <= end:
                f_out.write(line + '\n')

def extract_appropriate_freebase(train_file, kb_file):
    train = Text(train_file)
    # all entities in train set
    train_entities = train.entity_set
    facts = []
    print('Reading kb file at {}'.format(kb_file))
    kb_file_out = open(kb_file+'.small', 'w')
    with open(kb_file) as fb:
        for line in tqdm(fb):
            line = line.strip()
            line = line[1:-1]
            e1, r1, r2, e2 = [a.strip('"') for a in [x.strip() for x in line.split(',')]]
            if e1.strip() in train_entities or e2.strip() in train_entities:
                kb_file_out.write(line+'\n')

def count_num_entities(train_file, dev_file):
    num_entities = 0
    train = Text(train_file)
    dev = Text(dev_file)
    num_entities = train.num_entities + dev.num_entities
    print(train.num_entities, dev.num_entities, num_entities)
    train_entities = train.entity_set
    dev_entities = dev.entity_set
    print('Number of new elements in dev set {}'.format(len(dev_entities.difference(train_entities))))
    train_answer_entities = train.answer_entities
    dev_answer_entities = dev.answer_entities
    unseen_count = 0
    for entity in dev_answer_entities:
        if entity not in train_entities:
            unseen_count += 1
    print('Number of dev questions which has unseen entities as answer {}'.format(unseen_count))

def freebase_sort_wrt_entity1(freebase_file, output_file):
    facts_list = defaultdict(list)
    print('Reading kb file at {}'.format(freebase_file))
    with open(freebase_file) as fb:
        for line in tqdm(fb):
            line = line.strip()
            line = line[1:-1]
            e1, r1, r2, e2 = [a.strip('"') for a in [x.strip() for x in line.split(',')]]
            facts_list[e1].append([e1, r1, r2, e2])
    print('Writing to file...')
    f_out = open(output_file, 'w')
    for e1, facts_of_e1 in tqdm(facts_list.iteritems()):
        for facts in facts_of_e1:
            f_out.write(str(facts)+'\n')

def make_text_kb(train_file):
    """
    Make textual kb from sentences in training data
    :return:
    """
    output_file = "/home/rajarshi/canvas/data/SPADES_NEW/text_kb.spades.txt"
    if os.path.exists(output_file):
        os.remove(output_file)
    out = open(output_file, 'a')
    with open(train_file) as f_in:
        entity_to_sentence = defaultdict(list)
        print('Processing the training file...')
        for counter, line in tqdm(enumerate(f_in)):
            line = line.strip()
            qa = QuestionAnswer(line)  # parse each line
            question_words_list = qa.parsed_question['question'].split(' ')
            question_entities = qa.parsed_question['entities']
            question_entity_indices = qa.parsed_question['indices']
            for counter, index in enumerate(question_entity_indices): #replace words with entities
                question_words_list[index] = question_entities[counter]
            answer_entity = qa.parsed_question['answers'][0]
            #replace _blank_ in question_words_list
            answer_index = -1
            for counter, word in enumerate(question_words_list):
                if word == '_blank_':
                    question_words_list[counter] = answer_entity
                    answer_index = counter
            question_entities.append(answer_entity) #question_entities now contains all entities including answer entity
            question_entity_indices.append(answer_index) #question_entity_indices now contains all entity indices
            for question_entity in question_entities:
                entity_to_sentence[question_entity].append((question_words_list, question_entity_indices))
        print('Processing the entities in the train file...')
        for entity, sentence_index_list in tqdm(entity_to_sentence.iteritems()):
            text_kb_instance_map = {"entity":entity}
            for sentence_index_tuple in sentence_index_list:
                sentence, indices = sentence_index_tuple
                for index in indices:
                    if sentence[index] == entity:
                        continue
                    text_kb_instance_map['value'] = sentence[index]
                    sentence[index] = '_blank_'
                    text_kb_instance_map['key'] = sentence #key is the sentence
                    text_kb_instance_map['key_length'] = len(sentence)
                    #write the json into a file
                    text_kb_instance_json = json.dumps(text_kb_instance_map)
                    out.write(text_kb_instance_json+'\n')
                    sentence[index] = text_kb_instance_map['value']  # restoring it back


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--make_vocab', default=0, type=int)
    parser.add_argument('-v', '--vocab_dir', required=True)
    parser.add_argument('-t', '--train_file', required=True)
    parser.add_argument('-d', '--dev_file', required=True)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('-k', '--kb_file', required=True)
    parser.add_argument('--text_kb_file', default='', type=str)
    parser.add_argument('--extract_relevant_kb', default=0, type=int)
    parser.add_argument('--make_text_kb', default=0, type=int)
    parser.add_argument('--augment_text_kb_facts', default=0, type=int)
    parser.add_argument('--augment_kb_facts', default=0, type=int)
    parser.add_argument('--sort_freebase', default=0, type=int)

    args = parser.parse_args()
    make_vocab = (args.make_vocab == 1)
    extract_relevant_kb = (args.extract_relevant_kb == 1)
    train_file = args.train_file
    kb_file = args.kb_file
    text_kb_file = args.text_kb_file
    dev_file = args.dev_file
    test_file = args.test_file
    vocab_dir = args.vocab_dir
    create_text_kb = (args.make_text_kb == 1)
    augment_text_kb_facts = (args.augment_text_kb_facts == 1)
    augment_kb_facts = (args.augment_kb_facts == 1)
    sort_freebase = (args.sort_freebase == 1)

    if make_vocab:
        print('Creating entity and relation vocab')
        kb = KB(kb_file, create_vocab=True, vocab_dir=vocab_dir)
        print('Augmenting entity vocab with question words')
        text_qs = Text(train_file, kb=kb)
        text_qs.augment_to_entity_vocab()
        print('Saving...')
        kb.save_vocab(num_words=text_qs.num_words)
    elif extract_relevant_kb:
        print("Extracting KB triples bases on train set")
        extract_appropriate_freebase(train_file, kb_file)
    elif create_text_kb:
        print('Constructing text kb')
        input_file = "/home/rajarshi/canvas/data/SPADES_NEW/text.spades.txt.input"
        make_text_kb(input_file)
    elif sort_freebase:
        print('Sorting freebase wrt entity 1')
        freebase_file = "/home/rajarshi/canvas/data/TextKBQA/freebase.spades.txt.orig"
        output_file = "/home/rajarshi/canvas/data/TextKBQA/freebase.spades.txt.new"
        freebase_sort_wrt_entity1(freebase_file, output_file)
    elif augment_kb_facts:
        print('Augmenting files with kb facts')
        kb = KB(kb_file, vocab_dir=vocab_dir)
        print('Augmenting train file')
        train_file_out = "/iesl/canvas/rajarshi/data/SPADES/train_with_kb_facts.json"
        augment_qa_with_kb_facts(train_file, train_file_out, kb)
        print('Augmenting dev file')
        dev_file_out = "/iesl/canvas/rajarshi/data/SPADES/dev_with_kb_facts.json"
        augment_qa_with_kb_facts(dev_file, dev_file_out, kb)
        print('Augmenting test file')
        test_file_out = "/iesl/canvas/rajarshi/data/SPADES/test_with_kb_facts.json"
        augment_qa_with_kb_facts(test_file, test_file_out, kb)

    elif augment_text_kb_facts:
        print('Augmenting files with text kb facts')
        text_kb = TextKb(text_kb_file, vocab_dir=vocab_dir)
        print('Augmenting train file')
        train_file_out = "/iesl/canvas/rajarshi/data/SPADES_NEW/train_with_kb_and_text_facts.json"
        augment_qa_with_text_kb_facts(train_file, train_file_out, text_kb, is_train_file=True)
        print('Augmenting dev_file')
        dev_file_out = "/iesl/canvas/rajarshi/data/SPADES_NEW/dev_with_kb_and_text_facts.json"
        augment_qa_with_text_kb_facts(dev_file, dev_file_out, text_kb)
        print('Augmenting test_file')
        test_file_out = "/iesl/canvas/rajarshi/data/SPADES_NEW/test_with_kb_and_text_facts.json"
        augment_qa_with_text_kb_facts(test_file, test_file_out, text_kb)

        print('Done')

